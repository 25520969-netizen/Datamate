# -*- coding: utf-8 -*-
"""
Module 1: Ingestion & Schema Standardization Engine.
- Đọc và chuẩn hóa dữ liệu từ `.xlsx` (1 sheet, nhiều sheet theo tháng, hoặc nhiều sheet đa bảng),
  `.csv`, và `.sqlite` của từng doanh nghiệp.
- Chuẩn hóa toàn bộ tên bảng, tên cột và chuỗi về Unicode NFC (tránh lỗi NFD trên macOS).
- Tự động gộp các bảng cùng cấu trúc bị chia theo kỳ/đợt.
- Áp dụng quy tắc Mục 8 Rulebook:
  * Fill-down các thông tin cấp đơn hàng (trạng thái, khách hàng, ngày, giờ, ghi chú) bị để trống ở các dòng sau.
  * Loại bỏ hai dòng của cùng một đơn giống nhau ở mọi ô (chỉ áp dụng cho bảng có cột mã đơn;
    tuyệt đối giữ nguyên nếu bảng không ghi theo đơn).
  * Điền năm của sheet cho các ngày không ghi năm (`DD/MM`).
- Lưu cache dạng Parquet để truy vấn qua DuckDB / Pandas với tốc độ cao.
"""

import json
import re
import sqlite3
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from src.lexicon import load_lexicon

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass


ORDER_ID_CANDIDATES = {
    "Mã đơn hàng",
    "Mã đơn",
    "mã đơn",
    "Mã hóa đơn",
    "Mã phiếu",
    "Số phiếu",
    "Số đơn",
    "Mã GD",
    "Mã chứng từ",
    "Mã bill",
    "Ma don",
    "Số CT",
    "Số HĐ",
    "So hoa don",
    "Mã giao dịch",
}

ORDER_ID_FALLBACK_CANDIDATES = (
    "STT",
    "Số TT",
    "Số thứ tự",
    "Biển số",
    "Biển số xe",
    "IMEI",
)

ORDER_LEVEL_COL_PREFIXES = (
    "ngày",
    "ngay",
    "thời gian",
    "trạng thái",
    "trang_thai",
    "status",
    "kết quả",
    "xử lý",
    "tình trạng",
    "ghi chú",
    "ghi chu",
    "khách",
    "tên khách",
    "tenkh",
    "đối tác",
    "bên mua",
    "giờ",
    "kênh",
    "người bán",
    "phương thức",
    "hình thức",
    "trả bằng",
)


def to_nfc(text: Any) -> Any:
    """Chuẩn hóa chuỗi sang Unicode NFC, loại bỏ BOM \ufeff và gộp khoảng trắng ngang thừa."""
    if isinstance(text, str):
        cleaned = text.replace("\ufeff", "")
        s = unicodedata.normalize("NFC", cleaned).strip()
        return re.sub(r"[ \t]+", " ", s)
    return text


_DOT_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+$")
_PLAIN_INT_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_COMMA_RE = re.compile(r"^([+-]?\d+),(\d{1,2})$")
MONEY_COL_HINTS = ("tiền", "tien", "giá", "gia", "doanh thu", "nợ", "thanh toán", "phải trả", "phải thu", "chi phí", "vnd", "vnđ", "trị giá")


def _column_uses_dot_thousands(col_name: str, raw_values: pd.Series) -> bool:
    """
    "55.000" trong cột tiền là 55 nghìn đồng (dấu chấm phân cách nghìn kiểu Việt Nam), không phải 55.0.
    Chỉ kết luận theo CẢ CỘT: mọi ô là số nguyên hoặc dạng d.ddd(.ddd)*, có ít nhất một ô có dấu chấm,
    và (tên cột là tiền/giá hoặc có ô nhiều nhóm như "1.234.567").
    """
    vals = raw_values.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return False
    dotted = vals[vals.str.contains(".", regex=False)]
    if dotted.empty:
        return False
    if not vals.map(lambda v: bool(_DOT_THOUSANDS_RE.match(v) or _PLAIN_INT_RE.match(v))).all():
        return False
    is_money = any(h in str(col_name).lower() for h in MONEY_COL_HINTS)
    return is_money or int(dotted.str.count(r"\.").max()) >= 2


def _read_csv_robust(file_path: Path) -> pd.DataFrame:
    """Đọc file CSV với khả năng tự động xử lý BOM (utf-8-sig) và fallback đa bảng mã (windows-1258, latin-1)."""
    encodings_to_try = ["utf-8-sig", "utf-8", "windows-1258", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception) as e:
            last_err = e
            continue
        # pandas đã đọc "55.000" thành 55.0: đọc lại chuỗi gốc của các cột float để nhận phân cách nghìn
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        if float_cols:
            try:
                raw = pd.read_csv(file_path, encoding=enc, low_memory=False, dtype=str, usecols=float_cols)
                for c in float_cols:
                    if _column_uses_dot_thousands(c, raw[c]):
                        df[c] = raw[c].map(lambda v: float(v.strip().replace(".", "")) if isinstance(v, str) and v.strip() else None)
            except Exception:
                pass
        return df
    if last_err:
        raise last_err
    return pd.read_csv(file_path, low_memory=False)


def _accent_strip_key(text: Any) -> str:
    """Tạo khóa so khớp chuỗi không dấu, chữ thường, khoảng trắng đơn."""
    if not isinstance(text, str):
        text = str(text)
    s = unicodedata.normalize("NFC", text).strip()
    s = s.replace("đ", "d").replace("Đ", "d")
    s_ascii = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s_ascii.lower()).strip()


NON_NUMERIC_COL_KEYWORDS = (
    "mã",
    "ma_",
    "ma ",
    "id",
    "sku",
    "số phiếu",
    "số ct",
    "số hđ",
    "so hoa don",
    "phiếu thu",
    "điện thoại",
    "phone",
    "di động",
    "cccd",
    "cmnd",
    "mst",
    "mã số thuế",
    "barcode",
    "lô",
    "biển số",
    "ngày",
    "ngay",
    "thời gian",
    "thời điểm",
    "hsd",
    "date",
    "giờ",
    "tên",
    "ten",
    "nội dung",
    "trạng thái",
    "trang_thai",
    "status",
    "kết quả",
    "xử lý",
    "tình trạng",
    "đơn vị",
    "đvt",
    "dvt",
    "unit",
    "đ.vị",
    "nhóm",
    "nhà cung cấp",
    "ghi chú",
    "ghi chu",
    "diễn giải",
    "dien giai",
    "mô tả",
    "địa chỉ",
    "addr",
    "kênh",
    "phương thức",
    "hình thức",
    "trả bằng",
    "cách hoàn",
    "httt",
    "pttt",
    "người",
    "bàn",
    "khu vực",
    "hoạt chất",
    "loại",
    "thương hiệu",
    "thuộc tính",
    "vị trí",
    "nguồn",
    "mục",
    "product",
    "món",
    "hàng hóa",
    "hàng bán",
    "mặt hàng",
    "chi tiết hàng",
    "khách hàng",
    "khách",
    "đối tác",
    "bên mua",
    "hạng",
)

NUMERIC_FORCE_COL_KEYWORDS = (
    "tiền",
    "tien",
    "giá",
    "gia",
    "số lượng",
    "so_luong",
    "sluong",
    "slg",
    "sl",
    "tổng",
    "tong",
    "đã trả",
    "đã thu",
    "còn nợ",
    "chưa trả",
    "phải thu",
    "cần trả",
    "chi phí",
    "doanh thu",
    "tồn",
    "ton",
    "giảm",
    "ck",
    "chiết khấu",
    "tỉ lệ",
    "hệ số",
    "quy đổi",
    "bằng",
    "điểm",
    "số khách",
    "hạn nợ",
    "số ngày",
)

QTY_COL_KEYWORDS = (
    "số lượng",
    "so_luong",
    "sluong",
    "slg",
    "sl",
    "tồn",
    "ton",
)

ORDER_STATUS_COL_NAMES = {
    "trạng thái",
    "trạng thái đơn hàng",
    "trạng thái đơn",
    "trạng thái giao",
    "tình trạng giao",
    "tt đơn",
    "tt don",
    "trạng thái xử lý",
    "trang_thai",
    "status",
    "tình trạng",
    "tình trạng đơn",
    "xử lý",
    "kết quả",
    "ghi chú tt",
    "trangthai",
}

CANONICAL_STATUS_MAP = {
    # 1. Hoàn thành
    "hoàn thành": "Hoàn thành",
    "đã hoàn thành": "Hoàn thành",
    "thành công": "Hoàn thành",
    "xong": "Hoàn thành",
    "đã xong": "Hoàn thành",
    "đã lấy": "Hoàn thành",
    "đã giao": "Hoàn thành",
    "ok": "Hoàn thành",
    "completed": "Hoàn thành",
    # 2. Đã hủy
    "đã hủy": "Đã hủy",
    "đã huỷ": "Đã hủy",
    "đã hủy đơn": "Đã hủy",
    "đã huỷ đơn": "Đã hủy",
    "huỷ đơn": "Đã hủy",
    "hủy đơn": "Đã hủy",
    "hủy": "Đã hủy",
    "huỷ": "Đã hủy",
    "khách hủy": "Đã hủy",
    "khách huỷ": "Đã hủy",
    "không giao được": "Đã hủy",
    "cancel": "Đã hủy",
    "cancelled": "Đã hủy",
    # 3. Trả hàng
    "trả hàng": "Trả hàng",
    "đã trả hàng": "Trả hàng",
    "đã hoàn hàng": "Trả hàng",
    "khách trả": "Trả hàng",
    "hoàn trả": "Trả hàng",
    "đổi trả": "Trả hàng",
    "trả lại": "Trả hàng",
    "hàng trả về": "Trả hàng",
    "returned": "Trả hàng",
    # 4. Đang giao
    "đang giao": "Đang giao",
    "đang giao hàng": "Đang giao",
    "đang giao dịch": "Đang giao",
    "đang vận chuyển": "Đang giao",
    "đang ship": "Đang giao",
    "đang xử lý": "Đang giao",
    "chờ giao": "Đang giao",
    "đang gửi": "Đang giao",
    "shipping": "Đang giao",
}

_RETURN_STATUS_PAT = re.compile(r"trả hàng|trả lại|hoàn hàng|hoàn trả|đổi trả|khách trả|hàng trả|return", re.I)
_CANCEL_STATUS_PAT = re.compile(r"hủy|huỷ|cancel", re.I)
_SHIPPING_STATUS_PAT = re.compile(r"đang giao|đang vận chuyển|đang ship|chờ giao|chờ lấy|đang gửi|pending|shipping", re.I)
_DONE_STATUS_PAT = re.compile(r"hoàn thành|hoàn tất|thành công|đã giao|đã lấy|xong|completed|finish|delivered", re.I)
PAYMENT_STATUS_NAME_HINTS = ("thanh toán", "thanh_toan", "thanhtoan", "payment", "tt thanh", "công nợ", "cong no", "nợ")


def normalize_order_status(v: Any) -> Any:
    """Đưa trạng thái đơn về 4 giá trị chuẩn (Mục 8). Không nhận dạng được -> giữ nguyên (không đoán).
    Chỉ cụm "trả hàng/trả lại/hoàn hàng..." mới là trả hàng; "Đã trả" (đã trả tiền) KHÔNG phải trả hàng."""
    if not isinstance(v, str):
        return v
    key = v.strip().lower()
    if key in CANONICAL_STATUS_MAP:
        return CANONICAL_STATUS_MAP[key]
    lex_hit = load_lexicon().canonical_status(v)
    if lex_hit is not None:
        return lex_hit
    if _CANCEL_STATUS_PAT.search(key):
        return "Đã hủy"
    if _RETURN_STATUS_PAT.search(key):
        return "Trả hàng"
    if _SHIPPING_STATUS_PAT.search(key):
        return "Đang giao"
    if _DONE_STATUS_PAT.search(key):
        return "Hoàn thành"
    return v


def pick_order_status_col(df: pd.DataFrame) -> Optional[str]:
    """Chọn cột trạng thái ĐƠN (không phải trạng thái thanh toán): trong các cột có tên kiểu trạng thái,
    lấy cột có tỉ lệ giá trị nhận dạng được thành 4 trạng thái chuẩn cao nhất."""
    cands = []
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ("status_norm",) or any(h in cl for h in PAYMENT_STATUS_NAME_HINTS):
            continue
        if cl in ORDER_STATUS_COL_NAMES or any(cl.startswith(p) for p in ("trạng thái", "trang thai", "tt đơn", "tình trạng", "ghi chú tt")):
            cands.append(c)
    best, best_score = None, -1.0
    for c in cands:
        vals = df[c].dropna().astype(str)
        if vals.empty:
            continue
        mapped = vals.head(2000).map(normalize_order_status)
        score = mapped.isin(["Hoàn thành", "Đã hủy", "Trả hàng", "Đang giao"]).mean()
        if score > best_score:
            best, best_score = c, score
    return best


def _strict_name_key(text: Any) -> str:
    """Khóa so tên GIỮ dấu tiếng Việt: NFC, chữ thường, dấu câu -> khoảng trắng, gộp khoảng trắng."""
    s = unicodedata.normalize("NFC", str(text)).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()


_COMMA_NUM_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_SPACE_NUM_RE = re.compile(r"^[+-]?\d{1,3}(?:\s+\d{3})+(?:\.\d+)?$")
_DOT_VND_RE = re.compile(r"^([+-]?\d{1,3}(?:\.\d{3})+)\s*(?:đ|d|vnđ|vnd)?$", re.IGNORECASE)
_PLAIN_VND_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(?:đ|d|vnđ|vnd)$", re.IGNORECASE)
_K_VND_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*k$", re.IGNORECASE)
_TR_SPLIT_RE = re.compile(r"^([+-]?\d+)\s*tr\s*(\d+)$", re.IGNORECASE)
_TR_DEC_RE = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*tr$", re.IGNORECASE)
_QTY_UNIT_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s+[^\d].*$")
_DMY_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](20\d{2})(?:[T\s]+(\d{1,2}:\d{2}(?::\d{2})?))?$")
_VN_TEXT_DATE_RE = re.compile(
    r"^(\d{1,2})\s*(?:thg|tháng|thang)\s*(\d{1,2})(?:[,\s]+|\s+năm\s+|\s+nam\s+)(20\d{2})(?:[T\s]+(\d{1,2}:\d{2}(?::\d{2})?))?$",
    re.IGNORECASE,
)
_EXCEL_SERIAL_RE = re.compile(r"^\d{5}(?:\.\d+)?$")
_ISO_MIDNIGHT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T\s]+00:00(?::00)?(?:\.0+)?$")
CACHE_VERSION = "v16"


def parse_vn_numeric_value(val: Any, is_qty_col: bool = False) -> Optional[float]:
    """
    Chuyển đổi chuỗi số tiền/số lượng tiếng Việt ('90,000', '129 000', '55.000đ', '-53.500đ',
    '25k', '1tr440', '16,56tr', '1 bao 25kg', '2 xe 4m3') sang float chuẩn.
    """
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return None if pd.isna(val) else float(val)
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if _COMMA_NUM_RE.match(s):
        return float(s.replace(",", ""))
    m_dc = _DECIMAL_COMMA_RE.match(s)
    if m_dc:
        # "1,5" / "12,75": dấu phẩy thập phân kiểu Việt Nam (nhóm nghìn luôn đủ 3 chữ số)
        return float(f"{m_dc.group(1)}.{m_dc.group(2)}")
    if _SPACE_NUM_RE.match(s):
        return float(re.sub(r"\s+", "", s))
    m_dot = _DOT_VND_RE.match(s)
    if m_dot:
        return float(m_dot.group(1).replace(".", ""))
    m_plain = _PLAIN_VND_RE.match(s)
    if m_plain:
        return float(m_plain.group(1))
    m_k = _K_VND_RE.match(s)
    if m_k:
        return round(float(m_k.group(1).replace(",", ".")) * 1000.0, 4)
    m_tr1 = _TR_SPLIT_RE.match(s)
    if m_tr1:
        sign = -1.0 if m_tr1.group(1).startswith("-") else 1.0
        a = abs(int(m_tr1.group(1)))
        b = m_tr1.group(2)
        return round(sign * float(f"{a}.{b}") * 1_000_000.0, 4)
    m_tr2 = _TR_DEC_RE.match(s)
    if m_tr2:
        return round(float(m_tr2.group(1).replace(",", ".")) * 1_000_000.0, 4)
    if is_qty_col:
        m_q = _QTY_UNIT_RE.match(s)
        if m_q:
            return float(m_q.group(1))
    return None


def _excel_serial_to_iso(f_val: float) -> Optional[str]:
    if not (43831 <= f_val <= 47848):  # 2020-01-01 .. 2030-12-31
        return None
    from datetime import date, timedelta

    days_int = int(f_val)
    frac = f_val - days_int
    dt = date(1899, 12, 30) + timedelta(days=days_int)
    if frac > 1e-6:
        total_secs = int(round(frac * 86400))
        hh, rem = divmod(total_secs, 3600)
        mm, ss = divmod(rem, 60)
        if hh > 0 or mm > 0 or ss > 0:
            return f"{dt.isoformat()} {hh:02d}:{mm:02d}:{ss:02d}"
    return dt.isoformat()


def normalize_date_value_iso(val: Any) -> Any:
    """
    Chuẩn hóa giá trị ngày tháng dạng 'DD/MM/YYYY', '1 thg 6 2025', Excel serial ('45689'),
    hoặc 'YYYY-MM-DD 00:00:00' về chuẩn ISO 'YYYY-MM-DD' (giữ lại 'HH:MM:SS' nếu khác 00:00:00).
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    if hasattr(val, "isoformat"):
        iso_s = val.isoformat(sep=" ") if hasattr(val, "hour") else val.isoformat()
        m_mid = _ISO_MIDNIGHT_RE.match(iso_s)
        return m_mid.group(1) if m_mid else iso_s
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        conv = _excel_serial_to_iso(float(val))
        return conv if conv is not None else val
    if not isinstance(val, str):
        return val
    s = to_nfc(val).strip()
    if not s:
        return val
    m_mid = _ISO_MIDNIGHT_RE.match(s)
    if m_mid:
        return m_mid.group(1)
    if _EXCEL_SERIAL_RE.match(s):
        conv = _excel_serial_to_iso(float(s))
        if conv is not None:
            return conv
    for pattern in (_DMY_DATE_RE, _VN_TEXT_DATE_RE):
        m_d = pattern.match(s)
        if m_d:
            d_i, m_i, y_i = int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3))
            # Tự động phát hiện MM/DD/YYYY khi phần tử tháng > 12 (tháng hợp lệ chỉ từ 1 đến 12)
            if m_i > 12 and 1 <= d_i <= 12:
                d_i, m_i = m_i, d_i
            if 1 <= m_i <= 12 and 1 <= d_i <= 31:
                t_part = m_d.group(4)
                if t_part and not t_part.startswith("00:00"):
                    return f"{y_i:04d}-{m_i:02d}-{d_i:02d} {t_part}"
                return f"{y_i:04d}-{m_i:02d}-{d_i:02d}"
    return s


def _is_month_first_column(series: pd.Series) -> bool:
    """Cột ngày ghi MM/DD/YYYY: có ô phần thứ hai > 12 và KHÔNG có ô phần thứ nhất > 12.
    Khi đó mọi ô của cột (kể cả "03/04/2026") phải đọc theo tháng trước, không chỉ ô có ngày > 12."""
    if series.dtype != object:
        return False
    first_gt12 = second_gt12 = False
    for v in series.dropna().astype(str).head(5000):
        m = _DMY_DATE_RE.match(v.strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        first_gt12 = first_gt12 or a > 12
        second_gt12 = second_gt12 or b > 12
        if first_gt12:
            return False
    return second_gt12 and not first_gt12


def _normalize_month_first(val: Any) -> Any:
    if isinstance(val, str):
        m = _DMY_DATE_RE.match(to_nfc(val).strip())
        if m:
            val = f"{m.group(2)}/{m.group(1)}/{m.group(3)}" + (f" {m.group(4)}" if m.group(4) else "")
    return normalize_date_value_iso(val)


def coerce_formatted_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi các cột số học bị lưu dưới dạng chuỗi có dấu phẩy phân cách hàng nghìn,
    hậu tố 'đ', 'k', 'tr', hoặc số lượng kèm đơn vị ('1 bao 25kg', '2 xe 4m3') sang kiểu số.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue
        col_lower = col.lower().strip()
        is_forced_numeric = any(kw in col_lower for kw in NUMERIC_FORCE_COL_KEYWORDS) and not any(
            id_kw in col_lower
            for id_kw in (
                "mã",
                "ma_",
                "ma ",
                "id",
                "sku",
                "số phiếu",
                "số ct",
                "số hđ",
                "so hoa don",
                "phiếu thu",
                "điện thoại",
                "phone",
                "cccd",
                "cmnd",
                "mst",
                "mã số thuế",
                "barcode",
                "lô",
                "biển số",
                "ngày",
                "ngay",
                "thời gian",
                "thời điểm",
                "hsd",
                "date",
                "giờ",
                "trạng thái",
                "trang_thai",
                "status",
                "tình trạng",
                "xử lý",
                "kết quả",
                "ghi chú",
                "phương thức",
                "hình thức",
                "trả bằng",
                "cách hoàn",
                "tên",
                "ten",
            )
        )
        if not is_forced_numeric and any(kw in col_lower for kw in NON_NUMERIC_COL_KEYWORDS):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        non_empty = non_null[non_null.astype(str).str.strip() != ""]
        if non_empty.empty:
            continue
        is_qty_col = any(qkw in col_lower for qkw in QTY_COL_KEYWORDS)
        if _column_uses_dot_thousands(col, non_empty):
            df[col] = df[col].map(
                lambda v: float(str(v).strip().replace(".", "")) if isinstance(v, str) and str(v).strip() else (None if v is None or (isinstance(v, float) and pd.isna(v)) else v)
            )
            continue
        sample = non_empty.head(300)
        if sample.map(lambda v: parse_vn_numeric_value(v, is_qty_col=is_qty_col) is not None).all():
            converted = df[col].map(lambda v: parse_vn_numeric_value(v, is_qty_col=is_qty_col))
            if converted.notna().sum() == len(non_empty):
                # Nếu tất cả là số nguyên thì ép về int64 (nếu không có NaN) hoặc float64
                if converted.isna().sum() == 0 and (converted % 1 == 0).all():
                    df[col] = converted.astype("int64")
                else:
                    df[col] = converted.astype("float64")
    return df


def normalize_dataframe_nfc(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột, các cột kiểu chuỗi sang Unicode NFC, chuẩn hóa ngày ISO, trạng thái đơn và ép kiểu cột số."""
    df = df.copy()
    df.columns = [to_nfc(str(c)).strip() for c in df.columns]
    for col in df.columns:
        col_low = col.lower().strip()
        is_date_col = any(
            k in col_low for k in ("ngày", "ngay", "thời gian", "thời điểm", "hạn", "han ", "hsd", "date")
        ) and not any(ex in col_low for ex in ("số ngày", "so ngay", "hạn nợ"))
        if is_date_col:
            df[col] = df[col].map(normalize_date_value_iso if not _is_month_first_column(df[col]) else _normalize_month_first)
        elif df[col].dtype == object:
            df[col] = df[col].map(lambda x: to_nfc(x) if isinstance(x, str) else x)
        if col_low in ORDER_STATUS_COL_NAMES and df[col].dtype == object:
            df[col] = df[col].map(normalize_order_status)
    df = coerce_formatted_numeric_columns(df)
    return df


def extract_year_from_sheet_name(sheet_name: str) -> Optional[int]:
    """Trích xuất năm (2024..2029, hoặc hậu tố 2 chữ số -25, /25, _26) từ tên sheet nếu có."""
    m = re.search(r"\b(202[4-9])\b", sheet_name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(?:[/_-]|tháng\s*\d+\s*[/_-]?)(2[4-9])\b", sheet_name.lower())
    if m2:
        return 2000 + int(m2.group(1))
    return None


def fix_yearless_dates_in_series(series: pd.Series, sheet_year: Optional[int]) -> pd.Series:
    """
    Mục 8 Rulebook: 'Ngày không ghi năm: năm của sheet chứa dòng đó'.
    Nếu ô ngày có dạng 'DD/MM' hoặc 'D/M' (không có năm) và biết sheet_year, nối thêm '/{sheet_year}'.
    """
    if sheet_year is None or series.dtype != object:
        return series

    def _fix_val(val: Any) -> Any:
        if not isinstance(val, str):
            return val
        s = val.strip()
        if re.match(r"^\d{1,2}[/-]\d{1,2}$", s):
            sep = "/" if "/" in s else "-"
            return f"{s}{sep}{sheet_year}"
        return val

    return series.map(_fix_val)


LINE_LEVEL_NAME_HINTS = (
    "tên", "ten", "hàng", "hang", "sản phẩm", "san pham", "món", "mon", "sp", "sku", "barcode", "size",
    "đvt", "dvt", "đơn vị", "don vi", "quy cách", "topping", "lô", "hạn", "biến thể", "màu", "loại",
)


def _detect_order_constant_text_cols(
    df: pd.DataFrame, order_col: str, is_subtotal: pd.Series, exclude: set
) -> List[str]:
    """
    Mục 8 Rulebook: ô trống ở các dòng sau của một đơn giống ô dòng đầu. Ngoài các cột cấp đơn nhận theo tên,
    nhận thêm cột chữ (vd "Mã KH", "SĐT", "Thu ngân") mà dữ liệu cho thấy luôn bất biến trong mỗi đơn:
    ở >= 3 đơn nhiều dòng đã điền đủ, mọi dòng cùng giá trị; và có đơn bị để trống ở dòng sau.
    """
    cand: List[str] = []
    valid = df[order_col].notna() & ~is_subtotal
    for c in df.columns:
        if c == order_col or c in exclude or c.startswith("_") or df[c].dtype != object:
            continue
        c_low = c.lower()
        if any(h in c_low for h in LINE_LEVEL_NAME_HINTS):
            continue
        col = df.loc[valid, c].mask(df.loc[valid, c].astype(str).str.strip().isin(["", "nan", "<na>", "None"]))
        if not (col.isna().any() and col.notna().any()):
            continue
        grp = pd.DataFrame({"o": df.loc[valid, order_col], "v": col}).groupby("o")["v"]
        sizes = grp.size()
        full = grp.count()
        nuniq = grp.nunique()
        multi_full = (sizes >= 2) & (full == sizes)
        if multi_full.sum() >= 3 and (nuniq[multi_full] <= 1).all() and (nuniq <= 1).all():
            cand.append(c)
    return cand


def apply_rulebook_section8_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng quy tắc làm sạch dữ liệu tại Mục 8 Rulebook:
    1. Ô để trống ở các dòng sau của cùng một nhóm dòng: fill-down mã đơn và các thuộc tính cấp đơn
       (khách, ghi chú, trạng thái, ngày, giờ).
    2. Hai dòng của cùng một đơn giống nhau ở mọi ô: lỗi xuất file, tính một lần.
       (Chỉ áp dụng khi bảng có cột mã đơn; bảng không ghi theo đơn thì giữ nguyên mọi dòng).
    """
    if df.empty:
        return df

    df = df.copy()
    # Loại bỏ các dòng trống hoàn toàn
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        return df

    # Loại bỏ các dòng bị lặp lại tiêu đề cột (lỗi lặp header khi xuất file phân trang)
    if df.shape[1] >= 2 and len(df) >= 2:
        header_match = df.astype(str).apply(
            lambda col: col.str.strip().str.lower() == str(col.name).strip().lower()
        ).sum(axis=1)
        if (header_match >= 2).any():
            df = df[header_match < 2].reset_index(drop=True)
            if df.empty:
                return df

    # Tìm cột mã đơn (nếu có)
    order_col: Optional[str] = None
    for col in df.columns:
        if col in ORDER_ID_CANDIDATES:
            order_col = col
            break
    if order_col is None:
        for col in df.columns:
            if col in ORDER_ID_FALLBACK_CANDIDATES:
                order_col = col
                break

    if order_col is not None:
        # Thay thế chuỗi rỗng bằng NaN trên cột mã đơn để fill-down chính xác
        if df[order_col].dtype == object:
            df[order_col] = df[order_col].mask(df[order_col].astype(str).str.strip().isin(["", "nan", "<na>"]), pd.NA)

        # Phát hiện các dòng tổng kết / subtotal (ví dụ: Cộng ngày, Tổng tháng, Tổng cộng, Grand Total)
        # Các dòng này có mã đơn là NA và chứa từ khóa tổng kết ở một trong các cột văn bản.
        # Tuyệt đối KHÔNG fill-down mã đơn hoặc thông tin đơn hàng vào các dòng tổng kết này.
        subtotal_pat = re.compile(
            r"^\s*(?:cộng\s+ngày|tổng\s+tháng|tổng\s+cộng|grand\s+total|subtotal|cộng\b|tổng\b|tổng\s+phát\s+sinh|cộng\s+chuyển\s+sang|thành\s+tiền\s+cả\s+đợt|kế\s+toán\b|người\s+lập\b|bằng\s+chữ\b)",
            re.IGNORECASE,
        )
        is_subtotal = pd.Series(False, index=df.index)
        na_order_mask = df[order_col].isna()
        if na_order_mask.any():
            text_cols = [c for c in df.columns if c != order_col and df[c].dtype == object]
            for tc in text_cols:
                is_subtotal = is_subtotal | (na_order_mask & df[tc].astype(str).str.contains(subtotal_pat, na=False))

            # Phát hiện thêm các dòng footer / subtotal tự do không có từ khóa "tổng/cộng"
            # (dòng có số tiền nhưng mọi cột chiều như Ngày, Tên hàng, Khách hàng đều trống)
            dim_keywords = ("ngày", "ngay", "thời gian", "tên", "ten", "sản phẩm", "hàng hóa", "khách hàng", "khách", "mặt hàng", "món", "nội dung")
            present_dims = [c for c in df.columns if c != order_col and any(k in c.lower() for k in dim_keywords) and not c.startswith("_")]
            if present_dims:
                all_dims_na = df[present_dims].isna().all(axis=1) | df[present_dims].astype(str).isin(["", "None", "nan", "<NA>"]).all(axis=1)
                is_subtotal = is_subtotal | (na_order_mask & all_dims_na)

        # Chỉ fill-down mã đơn cho các dòng chi tiết đơn hàng thông thường (không phải subtotal)
        # Dòng subtotal giữ nguyên mã đơn là NA để không bị gộp chung vào đơn hàng cuối cùng
        if df[order_col].isna().any() and df[order_col].notna().any():
            s_order = df[order_col].copy()
            s_order_filled = s_order.where(~is_subtotal).ffill()
            df[order_col] = s_order_filled.where(~is_subtotal, pd.NA)

        # Gán trạng thái 'Subtotal' cho các dòng subtotal nếu bảng có cột trạng thái
        if is_subtotal.any():
            for status_c in df.columns:
                if status_c.lower() in ORDER_STATUS_COL_NAMES or any(status_c.lower().startswith(p) for p in ["trạng thái", "trang thai", "tt đơn", "status"]):
                    if df[status_c].dtype != object:
                        df[status_c] = df[status_c].astype(object)
                    df.loc[is_subtotal, status_c] = "Subtotal"

        # Chỉ thực hiện fill-down và lọc trùng dòng nếu một mã đơn xuất hiện trên nhiều dòng
        if not df[order_col].is_unique:
            order_level_cols = [
                c
                for c in df.columns
                if c != order_col and any(c.lower().startswith(p) for p in ORDER_LEVEL_COL_PREFIXES)
            ]
            order_level_cols += _detect_order_constant_text_cols(df, order_col, is_subtotal, exclude=set(order_level_cols))
            for c in order_level_cols:
                if df[c].dtype == object:
                    df[c] = df[c].mask(df[c].astype(str).str.strip().isin(["", "nan", "<na>"]), pd.NA)
                if df[c].isna().any() and df[c].notna().any():
                    src_order = df[order_col].where(df[c].notna() & ~is_subtotal).ffill()
                    df[c] = df[c].where(~is_subtotal).ffill().where((src_order == df[order_col]) & ~is_subtotal, df[c])

            # Loại bỏ hai dòng của cùng một đơn giống nhau ở mọi ô (chỉ áp dụng cho các dòng thuộc về một đơn hàng)
            data_cols = [c for c in df.columns if not c.startswith("_")]
            order_rows = df[df[order_col].notna() & ~is_subtotal].drop_duplicates(subset=data_cols, keep="first")
            no_order_rows = df[df[order_col].isna() & ~is_subtotal]
            df = pd.concat([order_rows, no_order_rows]).sort_index().reset_index(drop=True)
        elif is_subtotal.any():
            df = df[~is_subtotal].reset_index(drop=True)
    else:
        # Nếu bảng không có cột mã đơn, vẫn lọc triệt để các dòng subtotal/footer rác ở cuối bảng
        subtotal_pat = re.compile(
            r"^\s*(?:cộng\s+ngày|tổng\s+tháng|tổng\s+cộng|grand\s+total|subtotal|cộng\b|tổng\b|tổng\s+phát\s+sinh|cộng\s+chuyển\s+sang|thành\s+tiền\s+cả\s+đợt|kế\s+toán\b|người\s+lập\b|bằng\s+chữ\b)",
            re.IGNORECASE,
        )
        is_subtotal = pd.Series(False, index=df.index)
        text_cols = [c for c in df.columns if df[c].dtype == object and not c.startswith("_")]
        for tc in text_cols:
            is_subtotal = is_subtotal | df[tc].astype(str).str.contains(subtotal_pat, na=False)
        # Dòng tổng thật chỉ có nhãn "Tổng/Cộng..." và số; dòng nghiệp vụ như "Tổng vệ sinh máy lạnh | Sửa chữa | 2026-01-03"
        # còn nhiều ô chữ khác -> giữ lại (bảng không ghi theo đơn: mỗi dòng là một lần ghi)
        if text_cols and is_subtotal.any():
            filled_text = df[text_cols].apply(lambda col: col.notna() & (col.astype(str).str.strip() != ""))
            is_subtotal = is_subtotal & (filled_text.sum(axis=1) <= 1)

        dim_keywords = ("ngày", "ngay", "thời gian", "tên hàng", "sản phẩm", "hàng hóa", "khách hàng", "khách", "mặt hàng")
        present_dims = [c for c in df.columns if any(k in c.lower() for k in dim_keywords) and not c.startswith("_")]
        if present_dims:
            all_dims_na = df[present_dims].isna().all(axis=1) | df[present_dims].astype(str).isin(["", "None", "nan", "<NA>"]).all(axis=1)
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("_")]
            if num_cols:
                has_num = df[num_cols].notna().any(axis=1)
                is_subtotal = is_subtotal | (all_dims_na & has_num)

        if is_subtotal.any():
            df = df[~is_subtotal].reset_index(drop=True)

    return df


class BusinessDataStore:
    """Quản lý việc nạp, chuẩn hóa, lưu cache và trích xuất Schema cho từng doanh nghiệp."""

    def __init__(
        self,
        data_root: Path,
        cache_root: Optional[Path] = None,
    ) -> None:
        self.data_root = Path(data_root)
        if cache_root is not None:
            self.cache_root = Path(cache_root)
        elif Path("/content").exists():
            # Trên Google Colab, luôn ưu tiên SSD cục bộ /content/.cache thay vì ghi vào Drive qua FUSE
            self.cache_root = Path("/content/.cache/ingested")
        else:
            # Bố cục <tập>/data/<shop>: cache ở thư mục cha của tập; bố cục preprocessed <tập>/<shop>: cache cạnh thư mục tập
            base = self.data_root.parent.parent if self.data_root.name == "data" else self.data_root.parent
            self.cache_root = base / ".cache" / "ingested"
        self._mem_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._readme_cache: Dict[str, str] = {}
        self._date_range_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._category_cache: Dict[str, set] = {}
        self._schema_cols_cache: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self._schema_render_cache: Dict[Tuple[Any, ...], List[str]] = {}

    def get_biz_dir(self, business_id: str) -> Path:
        return self.data_root / business_id

    def get_readme(self, business_id: str) -> str:
        if business_id in self._readme_cache:
            return self._readme_cache[business_id]
        readme_path = self.get_biz_dir(business_id) / "README.md"
        if not readme_path.exists():
            readme_path = self.get_biz_dir(business_id) / "_readme.txt"
        content = to_nfc(readme_path.read_text(encoding="utf-8")) if readme_path.exists() else ""
        self._readme_cache[business_id] = content
        return content

    def get_shop_date_range(self, business_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Xác định ngày nhỏ nhất và lớn nhất có dữ liệu bán/nhập hàng của cửa hàng (tính một lần mỗi shop)."""
        if business_id not in self._date_range_cache:
            self._date_range_cache[business_id] = self._compute_shop_date_range(business_id)
        return self._date_range_cache[business_id]

    def get_category_values(self, business_id: str) -> set:
        """Giá trị chữ ngắn (bỏ dấu, chữ thường) của các cột ít giá trị (nhóm hàng, loại, kênh...) trong dữ liệu shop."""
        if business_id not in self._category_cache:
            vals: set = set()
            for df in self.load_tables(business_id).values():
                for col in df.columns:
                    if df[col].dtype != object:
                        continue
                    uniq = df[col].dropna().unique()
                    if 0 < len(uniq) <= 300:
                        vals.update(_accent_strip_key(v) for v in uniq if isinstance(v, str) and len(v) <= 60)
            self._category_cache[business_id] = vals
        return self._category_cache[business_id]

    def _compute_shop_date_range(self, business_id: str) -> Tuple[Optional[str], Optional[str]]:
        tables = self.load_tables(business_id)
        all_dts: List[str] = []
        for tname, df in tables.items():
            # Bỏ qua bảng thông tin tĩnh khách hàng, sản phẩm, bảng giá (chứa ngày sinh, ngày tạo sản phẩm không phải kỳ kinh doanh)
            if any(ex in tname.lower() for ex in ["khachhang", "khach_hang", "sanpham", "san_pham", "danhmuc", "danh_muc", "hanghoa", "hang_hoa", "banggia", "bang_gia"]):
                continue
            if "date_iso" in df.columns:
                s = df["date_iso"].dropna().astype(str)
                valid_dates = s[s.str.match(r"^\d{4}-\d{2}-\d{2}")]
                if len(valid_dates) > 0:
                    all_dts.append(valid_dates.min()[:10])
                    all_dts.append(valid_dates.max()[:10])
            for c in df.columns:
                if c == "date_iso":
                    continue
                if any(ex in c.lower() for ex in ["sinh", "birth", "tạo", "tao"]):
                    continue
                if any(k in c.lower() for k in ["ngày", "thời gian", "ngay", "date"]):
                    s = df[c].dropna().astype(str)
                    valid_dates = s[s.str.match(r"^\d{4}-\d{2}-\d{2}")]
                    if len(valid_dates) > 0:
                        all_dts.append(valid_dates.min()[:10])
                        all_dts.append(valid_dates.max()[:10])
        if all_dts:
            return (min(all_dts), max(all_dts))
        return (None, None)

    def _source_fingerprint(self, business_id: str) -> str:
        """Dấu vân tay thư mục nguồn: đường dẫn tuyệt đối + (tên, kích thước, mtime) của mọi file."""
        import hashlib

        biz_dir = self.get_biz_dir(business_id)
        h = hashlib.sha1(str(biz_dir.resolve()).encode("utf-8"))
        if biz_dir.exists():
            for f in sorted(biz_dir.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    h.update(f"{f.relative_to(biz_dir)}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
        return h.hexdigest()

    def load_tables(self, business_id: str, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Nạp toàn bộ bảng dữ liệu của một `business_id` thành dict[table_name, DataFrame].
        Tự động sử dụng Parquet cache nếu đã được chuẩn hóa trước đó.
        """
        if business_id in self._mem_cache:
            return self._mem_cache[business_id]

        biz_cache_dir = self.cache_root / business_id
        manifest_path = biz_cache_dir / "_manifest.json"

        fingerprint = self._source_fingerprint(business_id)
        if use_cache and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Cache chỉ hợp lệ cho đúng thư mục nguồn và đúng nội dung file (tránh tập test dùng nhầm cache
                # của tập train có cùng business_id, hoặc dữ liệu đã thay đổi)
                if manifest.get("version") == CACHE_VERSION and manifest.get("source_fingerprint") == fingerprint:
                    tables: Dict[str, pd.DataFrame] = {}
                    for tbl_name, pq_file in manifest.get("tables", {}).items():
                        pq_path = biz_cache_dir / pq_file
                        if pq_path.exists():
                            tables[to_nfc(tbl_name)] = pd.read_parquet(pq_path)
                    if tables:
                        tables = self._enrich_table_synonyms_and_id(tables, business_id)
                        tables = self._detect_and_unpivot_matrix_tables(tables)
                        self._mem_cache[business_id] = tables
                        return tables
            except Exception:
                pass

        tables = self._ingest_from_raw(business_id)

        if use_cache and tables:
            try:
                biz_cache_dir.mkdir(parents=True, exist_ok=True)
                manifest_tables: Dict[str, str] = {}
                for idx, (tbl_name, df) in enumerate(tables.items()):
                    pq_filename = f"tbl_{idx:02d}.parquet"
                    pq_path = biz_cache_dir / pq_filename
                    # Chuyển các cột object hỗn hợp sang chuỗi/kiểu tương thích để lưu Parquet an toàn
                    df_to_save = self._prepare_for_parquet(df)
                    df_to_save.to_parquet(pq_path, index=False)
                    manifest_tables[tbl_name] = pq_filename
                manifest_path.write_text(
                    json.dumps(
                        {"version": CACHE_VERSION, "business_id": business_id, "source_fingerprint": fingerprint, "tables": manifest_tables},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        tables = self._enrich_table_synonyms_and_id(tables, business_id)
        tables = self._detect_and_unpivot_matrix_tables(tables)
        self._mem_cache[business_id] = tables
        return tables

    @staticmethod
    def _enrich_table_synonyms_and_id(tables: Dict[str, pd.DataFrame], business_id: str) -> Dict[str, pd.DataFrame]:
        """Bổ sung cột business_id và các alias cột phổ biến cho mọi bảng để chống Binder Error trong DuckDB/Pandas."""
        for tbl_name, df in tables.items():
            synthetic = set()
            if "business_id" not in df.columns:
                df["business_id"] = business_id
                synthetic.add("business_id")

            # 1. Khách hàng (Tên)
            cust_name_syns = ["Khách hàng", "Tên KH", "Tên khách hàng", "Tên khách", "Khách hàng (tên)", "TenKH", "KH"]
            cust_name_syns = cust_name_syns + [n for n in load_lexicon().role_names("customer_name") if n not in cust_name_syns]
            found_cust_name = next((c for c in cust_name_syns if c in df.columns), None)
            if found_cust_name:
                for target in ["Khách hàng", "Tên KH", "Tên khách hàng", "Tên khách"]:
                    if target not in df.columns:
                        df[target] = df[found_cust_name]
                        synthetic.add(target)

            # 2. Mã khách hàng
            cust_id_syns = ["Mã KH", "Mã khách hàng", "Mã khách", "MaKH", "ma_kh", "Customer ID", "Mã khách mua", "MaKhachHang"]
            found_cust_id = next((c for c in cust_id_syns if c in df.columns), None)
            if found_cust_id:
                for target in ["Mã KH", "Mã khách hàng", "Mã khách"]:
                    if target not in df.columns:
                        df[target] = df[found_cust_id]
                        synthetic.add(target)

            # 3. Mã đơn hàng / Chứng từ
            order_syns = ["Mã đơn hàng", "Mã hóa đơn", "Mã chứng từ", "Số chứng từ", "SoCT", "Số HĐ", "Mã GD", "Mã đơn", "Ma don", "Số phiếu", "Mã giao dịch", "MaHoaDon", "MaDonHang"]
            order_syns = order_syns + [n for n in load_lexicon().role_names("order_id") if n not in order_syns]
            found_order = next((c for c in order_syns if c in df.columns), None)
            if not found_order:
                # Fallback: nếu bảng không có bất kỳ cột mã đơn chuẩn nào, nhưng có STT / Số TT / Biển số
                # (khi STT gom nhóm các dòng cùng đơn)
                fallback_order_syns = ["STT", "Số TT", "Số thứ tự", "Biển số", "Biển số xe", "IMEI"]
                found_order = next((c for c in fallback_order_syns if c in df.columns), None)
            if found_order:
                for target in ["Mã đơn hàng", "Mã hóa đơn", "Mã chứng từ", "Mã đơn"]:
                    if target not in df.columns:
                        df[target] = df[found_order]
                        synthetic.add(target)

            # 4. Tên sản phẩm / Hàng hóa
            prod_name_syns = ["Tên sản phẩm", "Tên hàng", "Tên hàng hóa", "Mặt hàng", "Tên mặt hàng", "Tên SP", "Sản phẩm", "ten_hang"]
            prod_name_syns = prod_name_syns + [n for n in load_lexicon().role_names("product_name") if n not in prod_name_syns]
            found_prod_name = next((c for c in prod_name_syns if c in df.columns), None)
            if found_prod_name:
                for target in ["Tên sản phẩm", "Tên hàng", "Tên hàng hóa", "Tên SP", "Sản phẩm"]:
                    if target not in df.columns:
                        df[target] = df[found_prod_name]
                        synthetic.add(target)

            # 5. Mã sản phẩm / SKU
            prod_id_syns = ["Mã sản phẩm", "Mã hàng", "Mã hàng hóa", "MaHangHoa", "MaHH", "Mã SP", "Mã SKU", "SKU", "ID SP", "Mã hàng bán"]
            found_prod_id = next((c for c in prod_id_syns if c in df.columns), None)
            if found_prod_id:
                for target in ["Mã sản phẩm", "Mã hàng", "Mã hàng hóa", "Mã SKU"]:
                    if target not in df.columns:
                        df[target] = df[found_prod_id]
                        synthetic.add(target)

            # 6. Ngày / Thời gian
            date_syns = ["Ngày", "Thời gian", "Ngay", "NgayLap", "Ngày tạo", "Ngày đặt hàng", "Ngày bán", "Ngày lập", "Ngày mua", "NgayBan", "Thời điểm", "Date"]
            date_syns = date_syns + [n for n in load_lexicon().role_names("date") if n not in date_syns]
            found_date = next((c for c in date_syns if c in df.columns), None)
            if found_date:
                for target in ["Ngày", "Thời gian", "NgàyLap"]:
                    if target not in df.columns:
                        df[target] = df[found_date]
                        synthetic.add(target)
                if "date_iso" not in df.columns:
                    s_dt = df[found_date].map(normalize_date_value_iso).astype(str)
                    df["date_iso"] = s_dt.str.extract(r"(\d{4}-\d{2}-\d{2})")[0].fillna(s_dt.str[:10])
                    synthetic.add("date_iso")
                if "order_date_iso" not in df.columns:
                    df["order_date_iso"] = df["date_iso"]
                    synthetic.add("order_date_iso")
                if "timestamp_iso" not in df.columns:
                    df["timestamp_iso"] = df[found_date].map(normalize_date_value_iso)
                    synthetic.add("timestamp_iso")

            # 7. Trạng thái đơn hàng
            status_syns = ["Trạng thái", "status_norm", "TinhTrang", "Tình trạng", "TrangThai", "Tình trạng đơn", "Trạng thái đơn hàng", "Trạng thái xử lý", "Xử lý"]
            status_syns = status_syns + [n for n in load_lexicon().role_names("status") if n not in status_syns]
            found_status = next((c for c in status_syns if c in df.columns), None)
            if found_status:
                for target in ["Trạng thái", "status_norm", "Tình trạng"]:
                    if target not in df.columns:
                        if target == "status_norm":
                            df[target] = df[found_status].map(normalize_order_status)
                        else:
                            df[target] = df[found_status]
                        synthetic.add(target)

            # 8. Doanh thu / Thành tiền / Tổng tiền / Trị giá
            revenue_syns = [
                "Thành tiền", "revenue_norm", "Tổng tiền", "ThanhTien", "thanh_tien", "thanhtien",
                "TongTien", "Tiền bán", "Tổng bill", "Tổng đơn", "Trị giá", "TriGia", "tri_gia", "Giá trị",
                "TongTienHang", "tong_tien_hang", "TongThanhToan", "tong_thanh_toan", "Tổng", "Tong",
                "Tổng cộng", "TongCong", "tiền hàng", "TienHang",
            ]
            revenue_syns = revenue_syns + [n for n in load_lexicon().role_names("amount") if n not in revenue_syns]
            found_rev = next((c for c in revenue_syns if c in df.columns), None)
            if found_rev:
                for target in ["Thành tiền", "revenue_norm", "Tổng tiền"]:
                    if target not in df.columns:
                        df[target] = df[found_rev]
                        synthetic.add(target)

            # 8b. Số lượng / Sản lượng / qty_base
            qty_syns = [
                "Số lượng", "qty_base", "SL", "Sluong", "Số Lượng", "Số lượng bán",
                "SL bán", "SLBan", "SoLuong", "Soluong", "Số lượng xuất", "SL xuất", "Số lượng lẻ", "SL lẻ"
            ]
            qty_syns = qty_syns + [n for n in load_lexicon().role_names("quantity") if n not in qty_syns]
            found_qty = next((c for c in qty_syns if c in df.columns), None)
            if found_qty:
                for target in ["Số lượng", "qty_base"]:
                    if target not in df.columns:
                        df[target] = df[found_qty]
                        synthetic.add(target)

            # 9. Nhóm hàng / Danh mục / Loại sản phẩm
            cat_syns = ["Nhóm hàng", "Nhóm", "Nhóm hàng hóa", "Danh mục", "Nhom hang", "Loại sản phẩm", "Loại hàng hóa", "Loại hàng", "Nhóm hàng(3 Cấp)"]
            cat_syns = cat_syns + [n for n in load_lexicon().role_names("category") if n not in cat_syns]
            found_cat = next((c for c in cat_syns if c in df.columns), None)
            if found_cat:
                for target in ["Nhóm hàng", "Nhóm", "Nhóm hàng hóa", "Danh mục", "Loại sản phẩm"]:
                    if target not in df.columns:
                        df[target] = df[found_cat]
                        synthetic.add(target)

            # 10. Hạn sử dụng (HSD)
            hsd_syns = ["Hạn sử dụng", "HSD", "HanSuDung", "Hạn dùng"]
            found_hsd = next((c for c in hsd_syns if c in df.columns), None)
            if found_hsd:
                for target in ["Hạn sử dụng", "HSD"]:
                    if target not in df.columns:
                        df[target] = df[found_hsd]
                        synthetic.add(target)

            existing_syn = df.attrs.get("synthetic_aliases", set())
            if isinstance(existing_syn, set):
                df.attrs["synthetic_aliases"] = existing_syn.union(synthetic)
            else:
                df.attrs["synthetic_aliases"] = synthetic

        return tables

    @staticmethod
    def _prepare_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                # Đảm bảo các giá trị datetime/time hoặc hỗn hợp trong cột object được đưa về string chuẩn
                out[col] = out[col].map(
                    lambda v: v.isoformat()
                    if hasattr(v, "isoformat")
                    else (str(v) if v is not None and not pd.isna(v) else None)
                )
        return out

    @staticmethod
    def _detect_and_unpivot_matrix_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Tự động phát hiện các bảng báo cáo ma trận 2 chiều (hàng là sản phẩm/khoản mục, cột là Tháng 1..12)
        và tạo thêm bảng unpivot song song dạng quan hệ phẳng ({table_name}_unpivot) vào DuckDB.
        """
        month_pat = re.compile(
            r"^(?:tháng\s*0?([1-9]|1[0-2])|t0?([1-9]|1[0-2])|0?([1-9]|1[0-2])[/-]20\d\d)$",
            re.IGNORECASE,
        )
        new_tables = dict(tables)
        for tname, df in tables.items():
            if tname.endswith("_unpivot"):
                continue
            month_cols = [c for c in df.columns if month_pat.match(to_nfc(str(c)).strip())]
            has_date_col = any(c.lower() in ["ngày", "thời gian", "ngay", "date"] for c in df.columns)
            if len(month_cols) >= 3 and not has_date_col:
                id_cols = [c for c in df.columns if c not in month_cols and not c.startswith("_")]
                if id_cols:
                    try:
                        df_melted = pd.melt(
                            df,
                            id_vars=id_cols,
                            value_vars=month_cols,
                            var_name="Thời gian",
                            value_name="Doanh thu",
                        )
                        df_melted["Ngày"] = df_melted["Thời gian"]
                        df_melted["Thành tiền"] = df_melted["Doanh thu"]
                        df_melted["revenue_norm"] = pd.to_numeric(
                            df_melted["Doanh thu"].astype(str).str.replace(r"[^\d.-]", "", regex=True),
                            errors="coerce",
                        ).fillna(0.0)
                        if "business_id" in df.columns:
                            df_melted["business_id"] = df["business_id"].iloc[0]
                        new_tables[f"{tname}_unpivot"] = df_melted
                    except Exception:
                        pass
        return new_tables

    def _ingest_from_raw(self, business_id: str) -> Dict[str, pd.DataFrame]:
        biz_dir = self.get_biz_dir(business_id)
        if not biz_dir.exists():
            return {}

        raw_tables: Dict[str, pd.DataFrame] = {}

        # Nếu thư mục là preprocessed có thư mục con tables/ (chứa các file CSV đã làm sạch 100%)
        tables_sub = biz_dir / "tables"
        if tables_sub.exists() and tables_sub.is_dir():
            # Mục 8 (bỏ dòng tổng, fill-down, bỏ dòng trùng của cùng một đơn) áp dụng cho dữ liệu nguồn csv/xlsx; nguồn sqlite
            # là cơ sở dữ liệu quan hệ, không áp dụng (giống nhánh đọc dữ liệu gốc). Dữ liệu preprocessed bản cũ có thể
            # chưa bỏ dòng trùng nên chạy lại ở đây (không đổi gì với dữ liệu đã sạch).
            apply_s8 = not self._readme_sources_sqlite_only(self.get_readme(business_id))
            for file_path in sorted(tables_sub.glob("*.csv")):
                stem = to_nfc(file_path.stem).strip()
                df = _read_csv_robust(file_path)
                df = normalize_dataframe_nfc(df)
                if apply_s8:
                    df = apply_rulebook_section8_cleaning(df)
                raw_tables[stem] = df
            # Dữ liệu preprocessed có thể chứa cả bảng gốc lẫn bảng gộp do bước tiền xử lý cũ sinh ra (vd bảng bán hàng
            # theo từng tháng + bảng cả năm): giữ một bản duy nhất để không bị đếm trùng
            before = dict(raw_tables)
            raw_tables = self._drop_redundant_union_tables(raw_tables)
            dropped = {n: len(before[n]) for n in before if n not in raw_tables}
            names_before_merge = set(raw_tables)
            raw_tables = self._merge_period_split_tables(raw_tables)
            # Bảng gộp mới tạo từ các bảng theo kỳ mang lại tên của bảng gộp cũ đã bỏ (nếu số dòng xấp xỉ) cho dễ hiểu
            for new_name in [n for n in raw_tables if n not in names_before_merge]:
                cands = [(n, r) for n, r in dropped.items() if n not in raw_tables and abs(r - len(raw_tables[new_name])) <= 0.1 * max(r, 1)]
                if cands:
                    best = max(cands, key=lambda x: x[1])[0]
                    raw_tables = {(best if k == new_name else k): v for k, v in raw_tables.items()}
            raw_tables = self._standardize_business_tables(raw_tables)
            raw_tables = self._apply_base_unit_conversion(raw_tables, self.get_readme(business_id))
            return raw_tables

        for file_path in sorted(biz_dir.iterdir()):
            if file_path.name.startswith(".") or file_path.name.lower() == "readme.md":
                continue

            suffix = file_path.suffix.lower()
            stem = to_nfc(file_path.stem).strip()

            if suffix == ".csv":
                df = _read_csv_robust(file_path)
                df = apply_rulebook_section8_cleaning(df)
                df = normalize_dataframe_nfc(df)
                raw_tables[stem] = df

            elif suffix == ".sqlite":
                conn = sqlite3.connect(str(file_path))
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tbl_names = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]
                    for t_name in tbl_names:
                        df = pd.read_sql_query(f'SELECT * FROM "{t_name}"', conn)
                        df = normalize_dataframe_nfc(df)
                        raw_tables[to_nfc(t_name)] = df
                finally:
                    conn.close()

            elif suffix == ".xlsx":
                readme_cols = {
                    to_nfc(m.strip())
                    for m in re.findall(r"-\s*\*\*(.+?)\*\*:", self.get_readme(business_id))
                }
                excel_tables = self._read_excel_smart(file_path, stem, readme_cols)
                raw_tables.update(excel_tables)

        raw_tables = self._merge_period_split_tables(raw_tables)

        raw_tables = self._standardize_business_tables(raw_tables)
        raw_tables = self._apply_base_unit_conversion(raw_tables, self.get_readme(business_id))
        return raw_tables

    @staticmethod
    def _readme_sources_sqlite_only(readme: str) -> bool:
        """README liệt kê các file nguồn dạng "- `tên` (loại)": True nếu mọi nguồn đều là sqlite."""
        kinds = re.findall(r"^\s*-\s*`[^`]+`\s*\(([A-Za-z0-9]+)\)", readme or "", flags=re.M)
        return bool(kinds) and all(k.lower() == "sqlite" for k in kinds)

    @staticmethod
    def _row_hashes(df: pd.DataFrame, cols: List[str]) -> "pd.Series":
        """Băm từng dòng trên các cột chung; số được làm tròn để 3 và 3.0 cho cùng giá trị băm."""
        parts = {}
        for c in cols:
            col = df[c]
            if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
                parts[c] = col.astype("float64").round(4).astype(str)
            else:
                parts[c] = col.astype(str).str.strip()
        return pd.util.hash_pandas_object(pd.DataFrame(parts), index=False)

    @classmethod
    def _drop_redundant_union_tables(cls, tables: Dict[str, pd.DataFrame], min_cover: float = 0.99) -> Dict[str, pd.DataFrame]:
        """
        Dữ liệu preprocessed có thể chứa bảng GỘP do bước tiền xử lý cũ sinh ra (bảng cả năm, bảng theo tháng gộp các
        năm...) bên cạnh các bảng gốc. Bảng gộp có thể sai lệch (vd có dòng trùng do gộp) và làm đếm trùng nếu giữ cả hai,
        nên giữ lại các bảng GỐC (mịn nhất) và bỏ bảng gộp.

        Nhận diện theo nội dung, không theo tên: bảng T là bảng gộp nếu >= `min_cover` các dòng (băm trên các cột chung có
        dữ liệu) của T nằm trong hợp các bảng NHỎ HƠN cùng cấu trúc, và các bảng nhỏ đó có tổng số dòng đủ lớn để chứa T.
        Duyệt từ bảng lớn đến nhỏ; bảng đã bị bỏ không còn được tính là nguồn.
        """
        def _nonempty_cols(df: pd.DataFrame) -> Dict[str, str]:
            return {str(c).lower().strip(): c for c in df.columns if not str(c).startswith("_") and df[c].notna().any()}

        names = sorted([n for n, d in tables.items() if len(d) >= 2], key=lambda n: -len(tables[n]))
        meta = {n: _nonempty_cols(tables[n]) for n in names}
        hash_cache: Dict[Tuple[str, Tuple[str, ...]], set] = {}

        def _hashes(name: str, cols: Tuple[str, ...]) -> set:
            key = (name, cols)
            if key not in hash_cache:
                sub = tables[name][[meta[name][c] for c in cols]]
                sub.columns = list(cols)
                hash_cache[key] = set(cls._row_hashes(sub, list(cols)).tolist())
            return hash_cache[key]

        removed: set = set()
        for t in names:
            parts_rows = 0
            n_parts = 0
            for sm in names:
                if sm == t or sm in removed or len(tables[sm]) >= len(tables[t]):
                    continue
                common = sorted(set(meta[t]) & set(meta[sm]))
                # bảng nhỏ phải cùng cấu trúc với bảng lớn: gần như mọi cột có dữ liệu của nó đều có ở bảng lớn
                if len(common) < 3 or len(common) < 0.8 * len(meta[sm]):
                    continue
                # So khớp trên cột SỐ / NGÀY / MÃ (ổn định); bỏ cột chữ tự do vì bước gộp bảng có thể đã chuẩn hoá lại
                # cách viết tên hàng/tên khách ở bảng gộp
                key_cols = [
                    c for c in common
                    if pd.api.types.is_numeric_dtype(tables[sm][meta[sm][c]])
                    or re.search(r"ngày|ngay|thời gian|date|mã|ma_|id|stt|imei|số phiếu", c)
                ]
                if len(key_cols) < 3:
                    continue
                cols = tuple(key_cols)
                small_h = _hashes(sm, cols)
                if small_h and len(small_h & _hashes(t, cols)) / len(small_h) >= min_cover:
                    parts_rows += len(tables[sm])
                    n_parts += 1
            # T là bản gộp nếu có >= 2 bảng nhỏ cùng cấu trúc nằm gọn trong T và tổng số dòng của chúng đủ chứa T
            if n_parts >= 2 and parts_rows >= 0.9 * len(tables[t]):
                removed.add(t)
        return {n: d for n, d in tables.items() if n not in removed}

    _PERIOD_TOKEN_RE = re.compile(
        r"(?:(?<=^)|(?<=[_\-\s]))(?:t|th|thg|thang|tháng|q|quy|quý|nam|năm|dot|đợt|ky|kỳ)?\s*\d{1,4}(?:[_\-]\d{2,4})?(?=$|[_\-\s])",
        re.I,
    )

    @classmethod
    def _merge_period_split_tables(cls, raw_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Gộp các bảng cùng cấu trúc bị chia theo kỳ/đợt (tên chỉ khác phần số: "X_1"/"X_2", "X_T1".."X_T12",
        "X_2025"/"X_2026", "T1-2025"...). Nhận diện theo tên bỏ phần kỳ + độ giống cột, không theo tên bảng cụ thể.
        Bảng mảnh được thay bằng bảng gộp (cột `_source_table` giữ nguồn).
        """
        def _jaccard(a: pd.Index, b: pd.Index) -> float:
            s1 = {str(c).lower().strip() for c in a if not str(c).startswith("_")}
            s2 = {str(c).lower().strip() for c in b if not str(c).startswith("_")}
            return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0

        groups: Dict[str, List[str]] = {}
        for name in raw_tables:
            base = cls._PERIOD_TOKEN_RE.sub("", to_nfc(name)).strip(" _-")
            if base != to_nfc(name).strip(" _-"):
                groups.setdefault(base, []).append(name)

        out = dict(raw_tables)
        for base, names in groups.items():
            if len(names) < 2:
                continue
            ref = raw_tables[names[0]].columns
            members = [n for n in names if _jaccard(ref, raw_tables[n].columns) >= 0.70]
            if len(members) < 2:
                continue
            target = base if base and base not in raw_tables else f"{base or 'bang'}_gop"
            parts = []
            for n in members:
                part = raw_tables[n].copy()
                part["_source_table"] = n
                parts.append(part)
                out.pop(n, None)
            out[target] = apply_rulebook_section8_cleaning(pd.concat(parts, ignore_index=True))
        return out

    # ------------------------------------------------------------------
    # Quy đổi về đơn vị cơ bản (Rulebook Mục 3 "Bán chạy", Mục 10 "Đơn vị cơ bản, đơn vị lớn")
    # ------------------------------------------------------------------
    _UNIT_COL_HINTS = ("đvt", "dvt", "đơn vị", "don vi", "don_vi", "unit", "đ.vị")
    _FACTOR_COL_HINTS = ("hệ số", "he so", "he_so", "quy đổi", "quy doi", "quy_doi", "factor", "số lượng quy", "sl quy", "quy cách")
    _FACTOR_COL_EXCLUDE = ("chiết khấu", "giảm", "thuế", "vat", "điểm", "%", "giá", "tiền")
    _PROD_NAME_COLS = ("tên hàng", "tên sản phẩm", "tên hàng hóa", "tên mặt hàng", "mặt hàng", "sản phẩm", "hàng hóa", "tên thuốc", "món", "ten_hang", "tên sp")
    _PROD_CODE_COLS = ("mã hàng", "mã sản phẩm", "mã sp", "mã hàng hóa", "mã thuốc", "sku", "mahang", "masp", "ma_sp", "mã mặt hàng")
    _QTY_COLS = ("số lượng", "sl", "so_luong", "soluong", "số lượng bán", "sl bán", "số lượng nhập", "sl nhập", "số lượng xuất")

    @classmethod
    def _find_col(cls, df: pd.DataFrame, names: Tuple[str, ...]) -> Optional[str]:
        low = {str(c).lower().strip(): c for c in df.columns}
        for n in names:
            if n in low:
                return low[n]
        return None

    @staticmethod
    def parse_readme_unit_conversions(readme: str) -> Dict[Tuple[Optional[str], str], Tuple[float, str, Optional[str]]]:
        """
        Đọc các dòng quy đổi trong README: "- Bia Tiger: 1 thùng = 4 lốc, 1 lốc = 6 lon" hoặc "1 thùng = 24 lon".
        Trả về {(khóa tên mặt hàng | None nếu áp chung, đơn vị lớn): (hệ số, đơn vị nhỏ hơn)}.
        """
        conv: Dict[Tuple[Optional[str], str], Tuple[float, str, Optional[str]]] = {}
        if not readme:
            return conv
        pair_pat = re.compile(r"1\s+([^\d=,;:()\n]+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+([^\d=,;:()\n.]+)", re.I)
        for line in to_nfc(readme).splitlines():
            pairs = list(pair_pat.finditer(line))
            if not pairs:
                continue
            head = line[: pairs[0].start()]
            item = re.sub(r"^[\s\-*•]+|\*\*", "", head).strip().rstrip(":").strip()
            item_key = _accent_strip_key(item) if item else None
            for m in pairs:
                big = _accent_strip_key(m.group(1))
                small = _accent_strip_key(m.group(3))
                try:
                    factor = float(m.group(2).replace(",", "."))
                except ValueError:
                    continue
                if big and small and big != small and factor > 0:
                    conv[(item_key, big)] = (factor, small, to_nfc(m.group(3)).strip())
        return conv

    @classmethod
    def _conversions_from_tables(cls, tables: Dict[str, pd.DataFrame]) -> Tuple[Dict[Tuple[Optional[str], str], Tuple[float, str, Optional[str]]], set]:
        """Bảng quy đổi / danh mục có cột hệ số: (mã hoặc tên hàng, đơn vị lớn) -> (hệ số, đơn vị cơ bản)."""
        conv: Dict[Tuple[Optional[str], str], Tuple[float, str, Optional[str]]] = {}
        conv_tables: set = set()
        for tname, df in tables.items():
            factor_col = next(
                (c for c in df.columns if any(h in str(c).lower() for h in cls._FACTOR_COL_HINTS)
                 and not any(x in str(c).lower() for x in cls._FACTOR_COL_EXCLUDE)
                 and pd.api.types.is_numeric_dtype(df[c])),
                None,
            )
            # Bảng quy đổi / danh mục không ghi theo đơn và không có cột ngày giao dịch
            is_transactional = any(c in ORDER_ID_CANDIDATES for c in df.columns) or any(
                any(k in str(c).lower() for k in ("ngày", "thời gian", "date")) for c in df.columns
            )
            if factor_col is None or is_transactional:
                continue
            unit_cols = [c for c in df.columns if any(h in str(c).lower() for h in cls._UNIT_COL_HINTS)]
            big_col = next((c for c in unit_cols if any(k in str(c).lower() for k in ("lớn", "lon", "quy đổi", "quy doi", "đóng gói"))), None)
            base_col = next((c for c in unit_cols if any(k in str(c).lower() for k in ("cơ bản", "co ban", "nhỏ", "lẻ", "le"))), None)
            if big_col is None:
                big_col = next((c for c in unit_cols if c != base_col), None)
            key_cols = [c for c in (cls._find_col(df, cls._PROD_CODE_COLS), cls._find_col(df, cls._PROD_NAME_COLS)) if c]
            if big_col is None or not key_cols:
                continue
            for _, row in df.iterrows():
                big = row.get(big_col)
                f = row.get(factor_col)
                if not isinstance(big, str) or pd.isna(f) or float(f) <= 0:
                    continue
                small = row.get(base_col) if base_col else None
                small_key = _accent_strip_key(small) if isinstance(small, str) else "__base__"
                small_disp = to_nfc(small).strip() if isinstance(small, str) else None
                for kc in key_cols:
                    kv = row.get(kc)
                    if isinstance(kv, str) and kv.strip():
                        conv[(_accent_strip_key(kv), _accent_strip_key(big))] = (float(f), small_key, small_disp)
            conv_tables.add(tname)
        return conv, conv_tables

    @classmethod
    def _apply_base_unit_conversion(cls, tables: Dict[str, pd.DataFrame], readme: str) -> Dict[str, pd.DataFrame]:
        """
        Mục 10: hệ số quy đổi đọc ở bảng quy đổi hoặc README; đơn vị không có dòng quy đổi là đơn vị cơ bản (hệ số 1).
        Với mọi bảng có (mặt hàng, đơn vị, số lượng) và chưa có cột số lượng cơ bản ghi sẵn:
          qty_base = số lượng × hệ số (hỗ trợ nhiều cấp: thùng -> lốc -> lon), thêm cột "Đơn vị cơ bản".
        """
        conv, conv_tables = cls._conversions_from_tables(tables)
        conv.update({k: v for k, v in cls.parse_readme_unit_conversions(readme).items() if k not in conv})
        if not conv:
            return tables
        explicit_base_names = {"sl lẻ", "số lượng lẻ", "quy ra đv lẻ", "sl quy đổi", "số lượng quy đổi", "sl cơ bản", "số lượng cơ bản", "quy_ra_dv_le"}

        def _resolve(keys: List[str], unit_key: str) -> Tuple[float, Optional[str]]:
            """Hệ số về đơn vị cơ bản và tên đơn vị cơ bản (None = không biết tên, chỉ biết hệ số)."""
            factor, u, disp = 1.0, unit_key, ""
            for _ in range(6):
                hit = next((conv[(k, u)] for k in keys if (k, u) in conv), None) or conv.get((None, u))
                if hit is None or hit[1] == u:
                    break
                factor *= hit[0]
                disp = hit[2]
                if hit[1] == "__base__":
                    break
                u = hit[1]
            return factor, (disp if factor != 1.0 else "")

        for tname, df in tables.items():
            if tname in conv_tables:
                continue
            if any(str(c).lower().strip() in explicit_base_names for c in df.columns):
                continue
            unit_col = next((c for c in df.columns if any(h in str(c).lower() for h in cls._UNIT_COL_HINTS)
                             and not any(k in str(c).lower() for k in ("giá", "gia", "cơ bản"))), None)
            qty_col = cls._find_col(df, cls._QTY_COLS)
            key_cols = [c for c in (cls._find_col(df, cls._PROD_CODE_COLS), cls._find_col(df, cls._PROD_NAME_COLS)) if c]
            if unit_col is None or qty_col is None or not key_cols or not pd.api.types.is_numeric_dtype(df[qty_col]):
                continue
            factors: List[float] = []
            base_units: List[Any] = []
            for _, row in df[key_cols + [unit_col]].iterrows():
                unit = row[unit_col]
                if not isinstance(unit, str) or not unit.strip():
                    factors.append(1.0)
                    base_units.append(unit)
                    continue
                keys = [_accent_strip_key(row[k]) for k in key_cols if isinstance(row[k], str)]
                f, base_name = _resolve(keys, _accent_strip_key(unit))
                factors.append(f)
                base_units.append(unit if f == 1.0 else base_name)
            if any(f != 1.0 for f in factors):
                df["qty_base"] = df[qty_col].astype(float) * pd.Series(factors, index=df.index)
                if "Đơn vị cơ bản" not in df.columns:
                    df["Đơn vị cơ bản"] = base_units
        return tables

    @classmethod
    def _spelling_canon_map(cls, values: List[str], preferred: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Gộp các cách viết của CÙNG một tên (Mục 3, 8): khác hoa/thường, khoảng trắng, dấu câu; hoặc bản gõ
        không dấu ("banh mi bo") khi chỉ có DUY NHẤT một bản có dấu tương ứng.
        KHÔNG gộp hai bản khác dấu ("Bánh mì bơ" vs "Bánh mì bò", "Anh Bảo" vs "Anh Bào"): khác dấu là khác chữ.
        Trả về {giá trị gốc: tên chuẩn} chỉ cho các giá trị cần đổi.
        """
        preferred = preferred or {}
        groups: Dict[str, List[str]] = {}
        for v in values:
            groups.setdefault(_strict_name_key(v), []).append(v)
        for sk in preferred:
            groups.setdefault(sk, [])
        canon: Dict[str, str] = {
            sk: preferred.get(sk) or cls._choose_best_canonical_spelling(vs) for sk, vs in groups.items()
        }
        accented_by_ascii: Dict[str, set] = {}
        for sk in groups:
            if not sk.isascii():
                accented_by_ascii.setdefault(_accent_strip_key(sk), set()).add(sk)
        out: Dict[str, str] = {}
        for sk, vs in groups.items():
            target = canon[sk]
            if sk.isascii():
                cands = accented_by_ascii.get(_accent_strip_key(sk), set())
                if len(cands) == 1:
                    target = canon[next(iter(cands))]
            for v in set(vs):
                if v != target:
                    out[v] = target
        return out

    @staticmethod
    def _choose_best_canonical_spelling(values: List[str]) -> str:
        """Chọn cách viết chuẩn nhất (có dấu tiếng Việt, không viết hoa toàn bộ) từ danh sách biến thể."""
        from collections import Counter

        counts = Counter(values)
        best_val = values[0]
        best_score = (-1, -1, -1)
        for v, freq in counts.items():
            has_non_ascii = 1 if any(ord(ch) > 127 for ch in v) else 0
            not_all_upper = 1 if (not v.isupper() and not v.islower()) else 0
            score = (has_non_ascii, not_all_upper, freq)
            if score > best_score:
                best_score = score
                best_val = v
        return best_val

    @classmethod
    def _standardize_business_tables(cls, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Chuẩn hóa liên bảng cho toàn bộ doanh nghiệp theo đúng Mục 1, 3, 6, 8 Rulebook:
        1. Chuẩn hóa tên mặt hàng (gộp các cách viết hoa/thường/thiếu dấu/khoảng trắng về tên chuẩn trong Danh mục).
        2. Chuẩn hóa tên khách lẻ ('kl', 'khách lẻ', 'khach le' -> 'Khách lẻ') và gộp cách viết tên khách ở shop không có mã khách.
        3. Chuẩn hóa ngày đặt của đơn nhiều dòng về min(ngày) của đơn đó, và lan truyền trạng thái 'Trả hàng'
           cho mọi dòng/bảng của đơn có trả hàng (kể cả trả 1 phần: Số lượng < 0, Số lượng trả > 0, bảng phiếu trả hàng).
        4. Bổ sung (denormalize) các cột tra cứu phổ biến (Thời gian, Trạng thái, Tên hàng, Hoạt chất, Thuốc kê đơn, Thành tiền)
           vào bảng chi tiết để tránh lỗi KeyError khi truy vấn trên 1 bảng.
        """
        # 1. Thu thập tên mặt hàng chuẩn từ các bảng danh mục / tồn kho (nếu có)
        catalog_name_map: Dict[str, str] = {}
        prod_col_candidates = (
            "Tên hàng",
            "Tên sản phẩm",
            "Tên hàng hóa",
            "Tên SP",
            "Mặt hàng",
            "Hàng bán",
            "Hàng",
            "Nội dung",
            "Product",
            "Món",
            "Chi tiết hàng",
            "Hàng hóa",
            "ten_sp",
            "TenHang",
        )
        # Bảng danh mục nhận theo CẤU TRÚC (không theo tên bảng cụ thể): có cột tên hàng, không có cột mã đơn,
        # và tên hàng gần như không lặp (mỗi dòng một mặt hàng)
        def _is_catalog(df_cat: pd.DataFrame) -> bool:
            if any(c in ORDER_ID_CANDIDATES for c in df_cat.columns):
                return False
            pc = next((c for c in prod_col_candidates if c in df_cat.columns and df_cat[c].dtype == object), None)
            if pc is None:
                return False
            vals = df_cat[pc].dropna()
            return len(vals) > 0 and vals.nunique() >= 0.9 * len(vals)

        for cat_tbl, df_cat in tables.items():
            if _is_catalog(df_cat):
                for pcol in prod_col_candidates:
                    if pcol in df_cat.columns and df_cat[pcol].dtype == object:
                        for v in df_cat[pcol].dropna().astype(str):
                            v_clean = to_nfc(v)
                            if v_clean:
                                catalog_name_map.setdefault(_strict_name_key(v_clean), v_clean)

        # Áp dụng chuẩn hóa tên mặt hàng và khách hàng trên mọi bảng
        cust_col_candidates = (
            "Khách hàng",
            "Tên khách hàng",
            "Khách",
            "TenKH",
            "Tên KH",
            "ten_kh",
            "Bên mua",
            "Đối tác",
            "TenKhachHang",
        )
        cust_id_candidates = ("Mã KH", "Mã khách hàng", "Mã KH mua", "Customer ID", "MaKH", "ma_kh")
        walkin_keys = {"kl", "khach le", "khach vang lai"}

        for tbl_name, df in tables.items():
            # Chuẩn hóa cột tên sản phẩm
            for pcol in prod_col_candidates:
                if pcol in df.columns and df[pcol].dtype == object:
                    non_null_vals = [str(x) for x in df[pcol].dropna().tolist() if str(x).strip()]
                    if not non_null_vals:
                        continue
                    col_canon_map = cls._spelling_canon_map(non_null_vals, preferred=catalog_name_map)
                    if col_canon_map:
                        df[pcol] = df[pcol].map(lambda x: col_canon_map.get(x, x) if isinstance(x, str) else x)

            # Chuẩn hóa cột tên khách hàng
            has_cust_id = any(cid in df.columns for cid in cust_id_candidates)
            for ccol in cust_col_candidates:
                if ccol in df.columns and df[ccol].dtype == object:
                    non_null_vals = [str(x) for x in df[ccol].dropna().tolist() if str(x).strip()]
                    if not non_null_vals:
                        continue
                    cust_canon_map: Dict[str, str] = {
                        v: "Khách lẻ" for v in set(non_null_vals) if _accent_strip_key(v) in walkin_keys and v != "Khách lẻ"
                    }
                    if not has_cust_id:
                        named = [v for v in non_null_vals if _accent_strip_key(v) not in walkin_keys]
                        cust_canon_map.update(cls._spelling_canon_map(named))
                    if cust_canon_map:
                        df[ccol] = df[ccol].map(lambda x: cust_canon_map.get(x, x) if isinstance(x, str) else x)

        # 2. Thu thập danh sách mã đơn có trả hàng (từ bảng lô, bảng trả hàng, dòng số lượng âm, hoặc dòng trạng thái Trả hàng)
        returned_order_ids: set = set()
        all_order_id_cols = set(ORDER_ID_CANDIDATES) | {"Order ID", "MaDon", "ma_don"}

        # Bảng phiếu trả hàng (nhận theo từ vựng trong tên bảng): mọi mã đơn gốc trong đó là đơn có trả hàng
        returns_ids: set = set()
        for tbl_name, df in tables.items():
            if not re.search(r"tra hang|trahang|tra lai|hoan hang|doi tra|return", _accent_strip_key(tbl_name).replace("_", " ")):
                continue
            ref_col = next(
                (c for c in df.columns
                 if re.search(r"ma ?don|ma ?hd|so ?hd|hoa don|order", _accent_strip_key(c).replace("_", " "))
                 and not re.search(r"phieu|tra", _accent_strip_key(c))),
                None,
            )
            if ref_col is not None:
                returns_ids.update(df[ref_col].dropna().astype(str))
        returned_order_ids.update(returns_ids)

        return_source_ids: set = set()
        for tbl_name, df in tables.items():
            ord_col = next((c for c in df.columns if c in all_order_id_cols), None)
            if not ord_col:
                continue
            n_before = len(returned_order_ids)
            # Kiểm tra cột số lượng trả > 0
            for c in df.columns:
                cl = c.lower()
                if ("lượng trả" in cl or "sl trả" in cl) and pd.api.types.is_numeric_dtype(df[c]):
                    ret_ids = df.loc[df[c] > 0, ord_col].dropna().astype(str).tolist()
                    returned_order_ids.update(ret_ids)
                elif any(q in cl for q in ("số lượng", "so_luong", "sluong", "slg", "sl")) and "trả" not in cl and "tồn" not in cl:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        neg_ids = df.loc[df[c] < 0, ord_col].dropna().astype(str).tolist()
                        returned_order_ids.update(neg_ids)
                elif cl in ORDER_STATUS_COL_NAMES and df[c].dtype == object:
                    st_ret_ids = df.loc[df[c] == "Trả hàng", ord_col].dropna().astype(str).tolist()
                    returned_order_ids.update(st_ret_ids)
            if len(returned_order_ids) > n_before:
                return_source_ids.update(df[ord_col].dropna().astype(str))
        return_source_ids.update(returns_ids)

        # Áp dụng chuẩn hóa ngày đặt (min date của đơn) và trạng thái 'Trả hàng' cho các đơn trong returned_order_ids
        order_date_col_candidates = ("Ngày giao dịch", "Ngày lập", "Ngày đặt hàng", "Ngày bán", "Ngày", "ngay_ban", "Thời gian", "Ngày mua")
        for tbl_name, df in tables.items():
            # Tạo alias cột ngày chuẩn ('Ngày', 'Ngày đặt hàng', 'Thời gian') trên mọi bảng có cột ngày giao dịch thật
            dt_found = next(
                (c for c in df.columns if any(k in c.lower() for k in ["ngày", "thời gian", "ngay", "date"]) and not any(k in c.lower() for k in ["sinh", "hạn", "số ngày", "so_ngay", "tuổi"])),
                None
            )
            if dt_found:
                for alias in ("Ngày", "Ngày đặt hàng", "Thời gian"):
                    if alias not in df.columns:
                        df[alias] = df[dt_found]
                # Chuẩn hóa cưỡng chế date_iso (YYYY-MM-DD), order_date_iso và timestamp_iso
                if "date_iso" not in df.columns:
                    s_dt = df[dt_found].map(normalize_date_value_iso).astype(str)
                    df["date_iso"] = s_dt.str.extract(r"(\d{4}-\d{2}-\d{2})")[0].fillna(s_dt.str[:10])
                if "order_date_iso" not in df.columns:
                    df["order_date_iso"] = df["date_iso"]
                if "timestamp_iso" not in df.columns:
                    df["timestamp_iso"] = df[dt_found].map(normalize_date_value_iso)

            # Nếu cột 'Thời gian' chỉ chứa ngày (không có giờ phút), nhưng có cột datetime chi tiết hơn, ưu tiên cột chi tiết
            if "Thời gian" in df.columns:
                detailed_dt = next(
                    (c for c in df.columns if c != "Thời gian" and any(k in c.lower() for k in ["thời gian", "thoi gian", "created", "datetime", "giờ", "gio"]) and df[c].astype(str).str.contains(r"\d{1,2}:\d{2}").any()),
                    None
                )
                if detailed_dt:
                    df["Thời gian"] = df[detailed_dt]

            # Tạo alias cột trạng thái chuẩn ('Trạng thái') trên mọi bảng có cột trạng thái
            st_found = pick_order_status_col(df)
            if st_found and "Trạng thái" not in df.columns:
                df["Trạng thái"] = df[st_found]

            if "status_norm" not in df.columns and (st_found or "Trạng thái" in df.columns):
                df["status_norm"] = df[st_found or "Trạng thái"].map(normalize_order_status)

            # 2c. Tạo alias cột tên sản phẩm chuẩn ('Tên hàng')
            if "Tên hàng" not in df.columns:
                item_found = next(
                    (c for c in df.columns if any(c.lower().strip() == p for p in [
                        "tên hàng", "tên sản phẩm", "tên hàng hóa", "tên mặt hàng", "hàng bán", "mặt hàng",
                        "hàng", "product", "nội dung", "sản phẩm", "ten_hang", "ten hang", "ten san pham"
                    ])),
                    None
                )
                if item_found:
                    df["Tên hàng"] = df[item_found]

            # 2d. Tạo alias cột số lượng quy đổi ra đơn vị cơ bản lẻ ('qty_base')
            if "qty_base" not in df.columns:
                base_qty_col = next(
                    (c for c in df.columns if any(c.lower().strip() == p for p in [
                        "qty_base", "sl lẻ", "số lượng lẻ", "quy ra đv lẻ", "sl quy đổi", "số lượng quy đổi",
                        "sl cơ bản", "số lượng cơ bản", "sl thực giao", "quy_ra_dv_le"
                    ])),
                    None
                )
                if base_qty_col:
                    df["qty_base"] = df[base_qty_col]
                else:
                    raw_qty_col = next(
                        (c for c in df.columns if any(c.lower().strip() == p for p in [
                            "số lượng", "so_luong", "soluong", "số lượng sản phẩm", "số lượng bán", "sl bán", "sl"
                        ])),
                        None
                    )
                    if raw_qty_col:
                        df["qty_base"] = df[raw_qty_col]

            # 2e. Tạo alias cột doanh thu thuần tiền hàng ('revenue_norm')
            if "revenue_norm" not in df.columns:
                pure_rev_col = next(
                    (c for c in df.columns if any(c.lower().strip() == p for p in [
                        "tien hang", "tiền hàng", "thành tiền", "thanh tien", "thanh_tien", "thanhtien",
                        "thành tiền (đ)", "amount", "tổng tiền hàng", "tongtienhang", "tong_tien_hang",
                        "trigia", "tri_gia", "trị giá",
                    ])),
                    None
                )
                if pure_rev_col:
                    df["revenue_norm"] = df[pure_rev_col]
                else:
                    tot_rev_col = next(
                        (c for c in df.columns if any(c.lower().strip() == p for p in [
                            "tổng tiền", "tongtien", "tongcong", "tổng cộng", "tiền", "thành tiền",
                            "tổng", "tong", "tổng đơn", "tong_don", "tổng bill", "tong_bill",
                            # "Tổng thanh toán" gồm VAT/phí ship, đã trừ giảm giá cả đơn -> chỉ dùng khi không còn cột nào khác
                            "tongthanhtoan", "tong_thanh_toan", "tổng thanh toán",
                        ])),
                        None
                    )
                    if tot_rev_col:
                        df["revenue_norm"] = df[tot_rev_col]

            ord_col = next((c for c in df.columns if c in all_order_id_cols), None)
            if not ord_col:
                continue
            # Nếu bảng có nhiều dòng trên cùng 1 mã đơn và có cột ngày đặt + cột trạng thái đơn 
            st_col = next((c for c in df.columns if c.lower().strip() in ORDER_STATUS_COL_NAMES), None)
            dt_col = next((c for c in df.columns if c in order_date_col_candidates), None)
            if dt_col and st_col and not df[ord_col].is_unique and df[dt_col].dtype == object:
                min_dates = df.groupby(ord_col, sort=False)[dt_col].min()
                df[dt_col] = df[ord_col].map(min_dates).fillna(df[dt_col])
            # Chỉ lan truyền sang bảng cùng "không gian mã đơn" với bảng phát hiện trả hàng (header/chi tiết của cùng đơn),
            # không sang phiếu nhập/phiếu chi chỉ vì trùng số mã ("15" của phiếu nhập khác đơn bán "15")
            tbl_ids = set(df[ord_col].dropna().astype(str))
            same_space = bool(tbl_ids) and len(tbl_ids & return_source_ids) >= 0.5 * len(tbl_ids)
            if st_col and returned_order_ids and same_space:
                # Chỉ cập nhật những đơn chưa phải 'Đã hủy' sang 'Trả hàng'
                mask_ret = df[ord_col].astype(str).isin(returned_order_ids) & (df[st_col] != "Đã hủy")
                if mask_ret.any():
                    df.loc[mask_ret, st_col] = "Trả hàng"
                    if "Trạng thái" in df.columns:
                        df.loc[mask_ret, "Trạng thái"] = "Trả hàng"
                    if "status_norm" in df.columns:
                        df.loc[mask_ret, "status_norm"] = "Trả hàng"

        # 3. Tự động liên kết khóa ngoại tổng quát (Generic Foreign Key & Catalog Denormalizer)
        # Hoàn toàn tự động dựa trên giao điểm khóa ngoại, KHÔNG hardcode tên bảng của shop cụ thể.
        all_order_id_set = set(all_order_id_cols) | {
            "Order ID", "MaDon", "ma_don", "Mã chứng từ", "MaHoaDon", "Mã đơn hàng", "Số phiếu", "Mã giao dịch", "Số CT", "Mã hóa đơn"
        }

        # 3a. Liên kết Bảng cấp Đơn (Header) -> Bảng chi tiết dòng (Detail) qua Mã đơn / Mã phiếu
        for parent_name, df_parent in list(tables.items()):
            parent_id_col = next((c for c in df_parent.columns if c in all_order_id_set), None)
            if not parent_id_col:
                continue

            parent_dedup = df_parent.drop_duplicates(subset=[parent_id_col]).set_index(parent_id_col)
            cols_to_share = [
                c for c in (
                    "Thời gian", "Ngày", "Trạng thái", "status_norm", "Mã KH", "Khách hàng", "Tên KH",
                    "Kênh bán", "Phương thức thanh toán", "Phương thức TT", "Người bán", "Ghi chú"
                )
                if c in parent_dedup.columns
            ]
            if not cols_to_share:
                continue

            for child_name, df_child in tables.items():
                if child_name == parent_name:
                    continue
                child_id_col = next(
                    (c for c in df_child.columns if c in all_order_id_set and (
                        c == parent_id_col or len(set(df_parent[parent_id_col].dropna().astype(str)) & set(df_child[c].dropna().astype(str))) > 0
                    )),
                    None
                )
                if child_id_col:
                    for col_name in cols_to_share:
                        if col_name not in df_child.columns:
                            df_child[col_name] = df_child[child_id_col].map(parent_dedup[col_name])

        # 3b. Liên kết Bảng Danh mục Hàng hóa / Thuốc -> Bảng bán / nhập qua Mã hàng / SKU
        prod_id_set = {
            "Mã hàng", "MaHang", "Mã SKU", "SKU", "MaHangHoa", "Mã sản phẩm", "Mã sản phẩm bán",
            "Mã mặt hàng", "Mã hàng bán", "ID SP", "MaSP", "ma_sp", "Ma hang", "Mã HH"
        }
        for cat_name, df_cat in list(tables.items()):
            cat_id_col = next((c for c in df_cat.columns if c in prod_id_set), None)
            if not cat_id_col:
                continue
            cat_dedup = df_cat.drop_duplicates(subset=[cat_id_col]).set_index(cat_id_col)
            attrs_to_share = [
                c for c in (
                    "Tên hàng", "Tên sản phẩm", "Tên SP", "Tên mặt hàng", "Mặt hàng", "Sản phẩm", "Món",
                    "Hoạt chất", "Thuốc kê đơn", "Kê đơn", "Nhóm hàng", "Nhom hang", "DanhMuc",
                    "Giá nhập", "Đơn giá", "Đơn vị tính", "ĐVT"
                )
                if c in cat_dedup.columns
            ]
            if not attrs_to_share:
                continue
            for target_name, df_target in tables.items():
                if target_name == cat_name:
                    continue
                target_id_col = next(
                    (c for c in df_target.columns if c in prod_id_set and (
                        c == cat_id_col or len(set(df_cat[cat_id_col].dropna().astype(str)) & set(df_target[c].dropna().astype(str))) > 0
                    )),
                    None
                )
                if target_id_col:
                    for attr in attrs_to_share:
                        if attr not in df_target.columns:
                            df_target[attr] = df_target[target_id_col].map(cat_dedup[attr])

        # 3c. Liên kết Bảng Khách hàng -> Bảng bán hàng qua Mã KH
        cust_id_set = {"Mã KH", "MaKhachHang", "Customer ID", "Mã khách mua", "ma_kh", "Số KH", "Mã khách lẻ/sỉ"}
        for cust_tbl_name, df_cust in list(tables.items()):
            cust_id_col = next((c for c in df_cust.columns if c in cust_id_set), None)
            if not cust_id_col:
                continue
            cust_name_col = next(
                (c for c in df_cust.columns if c != cust_id_col and c not in cust_id_set and any(k in c.lower() for k in ["tên", "ten", "tengoi", "bên mua"])),
                None
            )
            if not cust_name_col or cust_name_col not in df_cust.columns:
                continue
            cust_dedup = df_cust.drop_duplicates(subset=[cust_id_col]).set_index(cust_id_col)[cust_name_col]
            for target_name, df_target in tables.items():
                if target_name == cust_tbl_name:
                    continue
                target_cust_id = next((c for c in df_target.columns if c == cust_id_col or (c in cust_id_set and len(set(df_cust[cust_id_col].dropna().astype(str)) & set(df_target[c].dropna().astype(str))) > 0)), None)
                if target_cust_id and "Tên KH" not in df_target.columns and "Khách hàng" not in df_target.columns:
                    df_target["Tên KH"] = df_target[target_cust_id].map(cust_dedup)

        # Đảm bảo mọi bảng có 'Trạng thái' đều có 'status_norm', và mọi bảng có 'Ngày' đều có 'Thời gian'
        for tbl_name, df in tables.items():
            if "Trạng thái" in df.columns and "status_norm" not in df.columns:
                df["status_norm"] = df["Trạng thái"].map(normalize_order_status)
            if "Ngày" in df.columns and "Thời gian" not in df.columns:
                df["Thời gian"] = df["Ngày"]

        return tables

    @staticmethod
    def _parse_sheet_smart_header(
        xl: pd.ExcelFile,
        raw_s_name: str,
        readme_cols: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Đọc một sheet Excel và tự động phát hiện dòng tiêu đề cột thật sự nếu có các dòng
        tiêu đề báo cáo / dòng trống ở phía trên (gây ra cột 'Unnamed: ...').
        """
        df = xl.parse(raw_s_name)
        if df.empty and len(df.columns) == 0:
            return df

        col_strs = [to_nfc(str(c)).strip() for c in df.columns]
        has_unnamed = any(c.startswith("Unnamed:") or c == "nan" for c in col_strs)

        if has_unnamed:
            raw_df = xl.parse(raw_s_name, header=None)
            best_row_idx = -1
            best_score = -1
            max_scan = min(10, len(raw_df))
            for r_idx in range(max_scan):
                row_vals = [
                    to_nfc(str(v)).strip()
                    for v in raw_df.iloc[r_idx].tolist()
                    if pd.notna(v) and str(v).strip() != ""
                ]
                if len(row_vals) < max(2, int(raw_df.shape[1] * 0.5)):
                    continue
                # Dòng tiêu đề cột phải chứa chữ cái, không thể là dòng dữ liệu toàn số
                num_count = sum(1 for v in row_vals if re.match(r"^-?\d+(?:\.\d+)?$", v))
                if num_count >= len(row_vals) * 0.5:
                    continue
                # Tính điểm khớp với danh sách cột trong README.md + số lượng ô chuỗi không trùng
                readme_matches = sum(1 for v in row_vals if readme_cols and v in readme_cols)
                unique_strs = len(set(row_vals))
                score = readme_matches * 10 + unique_strs
                if score > best_score:
                    best_score = score
                    best_row_idx = r_idx

            if best_row_idx >= 0:
                header_vals = [
                    to_nfc(str(v)).strip() if pd.notna(v) and str(v).strip() != "" else f"col_{i}"
                    for i, v in enumerate(raw_df.iloc[best_row_idx].tolist())
                ]
                data_start_idx = best_row_idx + 1

                # Trường hợp 1: best_row_idx là dòng con, và best_row_idx - 1 là dòng cha (nhóm gộp)
                if best_row_idx > 0:
                    parent_row = raw_df.iloc[best_row_idx - 1].tolist()
                    parent_strs = [to_nfc(str(v)).strip() for v in parent_row if pd.notna(v) and str(v).strip() != ""]
                    if parent_strs and any(pd.isna(v) or str(v).strip() == "" for v in parent_row):
                        if all(not re.match(r"^-?\d+(?:\.\d+)?$", s) for s in parent_strs):
                            parent_filled = pd.Series(parent_row).ffill().tolist()
                            combined_header = []
                            for p_val, c_val in zip(parent_filled, header_vals):
                                p_clean = to_nfc(str(p_val)).strip() if pd.notna(p_val) else ""
                                c_clean = str(c_val).strip()
                                if p_clean and not c_clean.startswith("col_") and p_clean.lower() != c_clean.lower():
                                    combined_header.append(f"{p_clean} - {c_clean}")
                                elif c_clean and not c_clean.startswith("col_"):
                                    combined_header.append(c_clean)
                                elif p_clean:
                                    combined_header.append(p_clean)
                                else:
                                    combined_header.append(c_val)
                            header_vals = combined_header

                # Trường hợp 2: best_row_idx là dòng cha (có ô trống do merge) và dòng best_row_idx + 1 là dòng con
                elif best_row_idx + 1 < len(raw_df):
                    child_row = raw_df.iloc[best_row_idx + 1].tolist()
                    child_strs = [to_nfc(str(v)).strip() for v in child_row if pd.notna(v) and str(v).strip() != ""]
                    if len(child_strs) >= max(2, int(len(header_vals) * 0.5)) and all(not re.match(r"^-?\d+(?:\.\d+)?$", s) for s in child_strs):
                        parent_filled = pd.Series(raw_df.iloc[best_row_idx]).ffill().tolist()
                        combined_header = []
                        for i, (p_val, c_val) in enumerate(zip(parent_filled, child_row)):
                            p_clean = to_nfc(str(p_val)).strip() if pd.notna(p_val) else ""
                            c_clean = to_nfc(str(c_val)).strip() if pd.notna(c_val) else ""
                            if p_clean and c_clean and p_clean.lower() != c_clean.lower():
                                combined_header.append(f"{p_clean} - {c_clean}")
                            elif c_clean:
                                combined_header.append(c_clean)
                            elif p_clean:
                                combined_header.append(p_clean)
                            else:
                                combined_header.append(f"col_{i}")
                        header_vals = combined_header
                        data_start_idx = best_row_idx + 2

                df = raw_df.iloc[data_start_idx:].copy()
                df.columns = header_vals
                df = df.reset_index(drop=True)

        # Loại bỏ các dòng bị lặp lại chính dòng tiêu đề (khi nối hoặc xuất từ phần mềm)
        if not df.empty and len(df.columns) >= 2:
            first_col = df.columns[0]
            second_col = df.columns[1]
            dup_header_mask = (df[first_col].astype(str).str.strip() == str(first_col)) & (
                df[second_col].astype(str).str.strip() == str(second_col)
            )
            if dup_header_mask.any():
                df = df[~dup_header_mask].reset_index(drop=True)

        return df

    def _read_excel_smart(
        self,
        xlsx_path: Path,
        file_stem: str,
        readme_cols: Optional[set] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Đọc file Excel thông minh:
        - Nếu chỉ có 1 sheet -> trả về {file_stem: df}.
        - Nếu có nhiều sheet cùng cấu trúc cột (chia theo tháng) -> gộp thành 1 bảng {file_stem: df_merged},
          kèm xử lý điền năm từ tên sheet cho cột ngày không ghi năm.
        - Nếu các sheet có cấu trúc cột khác nhau 
          -> tách mỗi sheet thành một bảng riêng {sheet_name: df}.
        """
        xl = pd.ExcelFile(xlsx_path)
        sheet_dfs: List[Tuple[str, pd.DataFrame]] = []

        for raw_s_name in xl.sheet_names:
            s_name = to_nfc(raw_s_name).strip()
            df = self._parse_sheet_smart_header(xl, raw_s_name, readme_cols)
            if df.empty and len(df.columns) == 0:
                continue

            sheet_year = extract_year_from_sheet_name(s_name)
            if sheet_year is not None:
                for col in df.columns:
                    if any(k in col.lower() for k in ["ngày", "ngay", "thời gian", "date"]):
                        df[col] = fix_yearless_dates_in_series(df[col], sheet_year)

            df = apply_rulebook_section8_cleaning(df)
            df = normalize_dataframe_nfc(df)
            sheet_dfs.append((s_name, df))

        if not sheet_dfs:
            return {}

        if len(sheet_dfs) == 1:
            only_df = apply_rulebook_section8_cleaning(sheet_dfs[0][1])
            return {file_stem: only_df}

        # Kiểm tra độ tương đồng cấu trúc cột giữa các sheet bằng Jaccard Similarity (Schema-Similarity Consolidator)
        def _schema_jaccard(cols1, cols2) -> float:
            s1 = {str(c).lower().strip() for c in cols1 if not str(c).startswith("_")}
            s2 = {str(c).lower().strip() for c in cols2 if not str(c).startswith("_")}
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        # Gom các sheet cùng cấu trúc thành cụm (không phụ thuộc thứ tự sheet: "Danh mục" đứng đầu rồi 12 sheet tháng
        # vẫn gộp đủ 12 tháng). Cụm >= 2 sheet (phân mảnh theo tháng/đợt) -> một bảng; sheet lẻ giữ bảng riêng.
        clusters: List[List[Tuple[str, pd.DataFrame]]] = []
        for s_name, df in sheet_dfs:
            for cl in clusters:
                if _schema_jaccard(cl[0][1].columns, df.columns) >= 0.70:
                    cl.append((s_name, df))
                    break
            else:
                clusters.append([(s_name, df)])

        out: Dict[str, pd.DataFrame] = {}
        multi = [cl for cl in clusters if len(cl) >= 2]
        for cl in clusters:
            if len(cl) == 1:
                s_name, df = cl[0]
                out[s_name] = apply_rulebook_section8_cleaning(df)
                continue
            annotated_dfs: List[pd.DataFrame] = []
            for s_name, df in cl:
                df_copy = df.copy()
                df_copy["_sheet_name"] = s_name
                annotated_dfs.append(df_copy)
            merged_df = apply_rulebook_section8_cleaning(pd.concat(annotated_dfs, ignore_index=True))
            name = file_stem if len(multi) == 1 else f"{file_stem}_{cl[0][0]}"
            out[name] = merged_df
        return out

    def create_duckdb_connection(self, business_id: str) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, pd.DataFrame]]:
        """Tạo kết nối DuckDB in-memory đã đăng ký sẵn toàn bộ các bảng của `business_id`."""
        tables = self.load_tables(business_id)
        tables = self._enrich_table_synonyms_and_id(tables, business_id)
        con = duckdb.connect(database=":memory:")
        if sys.platform == "darwin":
            try:
                con.execute("PRAGMA threads=1; PRAGMA max_memory='1GB';")
            except Exception:
                pass
        for tbl_name, df in tables.items():
            con.register(tbl_name, df)
        return con, tables

    def build_compact_schema_prompt(self, business_id: str, max_sample_rows: int = 3, question: str = "") -> str:
        """
        Tạo Compact Schema Metadata + nội dung README.md của shop để đưa vào Prompt của LLM.
        Đảm bảo tổng độ dài gọn gàng (nằm gọn trong giới hạn 8192 token của vLLM cả khi retry).
        """
        # README giữ nguyên (không lọc theo câu hỏi) để phần đầu schema của một shop giống hệt nhau giữa các câu
        # -> vLLM dùng lại prefix cache. Quy đổi đơn vị đã được áp vào cột qty_base khi nạp dữ liệu.
        readme_text = self.get_readme(business_id).strip()

        tables = self.load_tables(business_id)

        # Bảng mảnh theo kỳ đã được thay bằng bảng gộp khi nạp; ẩn thêm mảnh "X_n" nếu bảng gốc "X" cùng tồn tại
        part_names_to_hide = set()
        for tbl_name in tables:
            m = re.match(r"^(.+?)_(\d+)$", tbl_name)
            if m and m.group(1).strip() in tables:
                part_names_to_hide.add(tbl_name)

        if len(readme_text) > 2400:
            readme_text = readme_text[:2400] + "\n...(xem thêm trong tài liệu shop)"

        lines = [
            "=== MÔI TRƯỜNG DỮ LIỆU: Toàn bộ bảng dưới đây đã được nạp sẵn vào kết nối DuckDB `con` và dict `tables`. KHÔNG mở kết nối mới ===",
            "",
            f"=== TÀI LIỆU MÔ TẢ DỮ LIỆU CỦA SHOP ({business_id}/README.md) ===",
            readme_text,
            "",
            f"=== DANH SÁCH BẢNG ĐÃ NẠP TRONG `tables` (Pandas dict) VÀ `con` (DuckDB) ===",
        ]

        def _infer_semantic_role(tbl_name: str, cols: List[str]) -> str:
            t_low = tbl_name.lower()
            c_str = " ".join(cols).lower()
            # 1. Bán hàng & Doanh thu (ưu tiên số 1 vì chứa số liệu cốt lõi)
            if any(k in t_low or k in c_str for k in ["bán", "ban_hang", "hóa đơn", "hoa_don", "đơn hàng", "don_hang", "thành tiền", "thanh tien", "chi_tiet"]):
                return "BẢNG BÁN HÀNG & DOANH THU"
            # 2. Công nợ & Sổ quỹ
            if any(k in t_low or k in c_str for k in ["nợ", "cong_no", "công nợ", "phiếu thu", "so_quy"]):
                return "BẢNG CÔNG NỢ & THU CHI"
            # 3. Tồn kho & Nhập hàng
            if any(k in t_low or k in c_str for k in ["tồn", "ton_kho", "nhập", "nhap_hang", "hạn dùng", "hsd"]):
                return "BẢNG TỒN KHO & NHẬP HÀNG"
            # 4. Danh mục sản phẩm
            if any(k in t_low or k in c_str for k in ["sản phẩm", "san_pham", "thuoc", "thuc_don", "danh mục"]):
                return "BẢNG DANH MỤC SẢN PHẨM"
            # 5. Danh mục khách hàng
            if any(k in t_low or k in c_str for k in ["khách hàng", "khach_hang", "đối tác"]):
                return "BẢNG THÔNG TIN KHÁCH HÀNG"
            return "BẢNG DỮ LIỆU HOẠT ĐỘNG"

        def _infer_col_type(col_name: str, series: pd.Series) -> str:
            # Khai báo đúng kiểu vật lý trong DuckDB (cột ngày là chuỗi ISO, không phải TIMESTAMP)
            if pd.api.types.is_float_dtype(series):
                return "DOUBLE"
            if pd.api.types.is_integer_dtype(series):
                return "BIGINT"
            if pd.api.types.is_bool_dtype(series):
                return "BOOLEAN"
            non_null = series.dropna()
            if len(non_null) > 0 and non_null.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.8:
                return "VARCHAR -- ngày ISO 'YYYY-MM-DD[ HH:MM:SS]', lọc bằng CAST(... AS DATE)"
            if len(non_null) > 0 and series.dtype == object:
                uniq = non_null.unique().tolist()
                # Cột phân loại ít giá trị (trạng thái, kênh, nhóm, hình thức TT): liệt kê đủ giá trị thật để lọc đúng chính tả
                if 1 < len(uniq) <= 8 and all(len(str(u)) <= 30 for u in uniq):
                    return f"VARCHAR -- giá trị: {uniq}"
            return "VARCHAR"

        # Nhóm các bảng có cùng cấu trúc cột để tránh lặp lại hàng chục lần (như các bảng chia theo tháng)
        by_cols: Dict[Tuple[str, ...], List[Tuple[str, pd.DataFrame]]] = {}
        for tbl_name, df in tables.items():
            if tbl_name in part_names_to_hide:
                continue
            ck = (business_id, tbl_name)
            if ck in self._schema_cols_cache:
                by_cols.setdefault(self._schema_cols_cache[ck], []).append((tbl_name, df))
                continue
            synthetic = df.attrs.get("synthetic_aliases", set())
            # Giữ lại các cột thực tế và cột chuẩn hóa cốt lõi (status_norm, qty_base, revenue_norm)
            # Ẩn bớt các alias tổng hợp nhân bản trong Schema Prompt để tránh vượt ngưỡng 8192 token gây lỗi 400 Bad Request
            core_norm = {"status_norm", "qty_base", "revenue_norm"}
            visible: List[str] = []
            probe = df.head(2000)
            for c in df.columns:
                if str(c).startswith("_") or (c in synthetic and c not in core_norm):
                    continue
                # Alias trùng y hệt một cột đã hiện (Ngày đặt hàng/Thời gian/date_iso... sao chép từ "Ngày") -> ẩn
                if c not in core_norm and any(probe[c].equals(probe[v]) for v in visible):
                    continue
                visible.append(c)
            cols_key = tuple(visible)
            self._schema_cols_cache[ck] = cols_key  # tính một lần mỗi bảng (phép so cột tốn ~0.3s trên bảng lớn)
            by_cols.setdefault(cols_key, []).append((tbl_name, df))

        # Khử nhiễu: Khi shop có nhiều bảng (> 5 nhóm), chỉ hiển thị chi tiết các bảng liên quan tới câu hỏi
        # để tránh làm loãng attention và giảm 50-60% tokens vô ích cho LLM.
        items_to_render = list(by_cols.items())
        omitted_table_names: List[str] = []
        if len(items_to_render) > 5 and question:
            q_low = question.lower()
            q_words = [w for w in re.split(r"\W+", q_low) if len(w) >= 2]

            def _score_grp(item: Tuple[Tuple[str, ...], List[Tuple[str, pd.DataFrame]]]) -> int:
                c_key, grp_list = item
                names = [g[0] for g in grp_list]
                names_str = " ".join(names).lower()
                cols_str = " ".join(c_key).lower()
                sc = 0
                # 1. Bán hàng & Doanh thu cốt lõi
                if any(k in names_str for k in ["bán", "ban_hang", "banhang", "hóa đơn", "hoa_don", "đơn hàng", "don_hang", "donhang"]):
                    sc += 30
                elif any(k in cols_str for k in ["thành tiền", "thanh tien", "mã đơn", "ma don"]):
                    sc += 20
                # 2. Công nợ & Sổ quỹ & Thanh toán
                if any(k in q_low for k in ["nợ", "công nợ", "cong no", "trả nợ", "thu nợ", "thanh toán", "sổ quỹ", "so_quy"]):
                    if any(k in names_str or k in cols_str for k in ["nợ", "thanh_toan", "cong_no", "so_quy", "thu", "chi"]):
                        sc += 40
                # 3. Tồn kho & Nhập hàng
                if any(k in q_low for k in ["tồn", "ton ", "kho", "nhập", "nhap", "hết hàng", "lô", "hạn dùng", "hsd", "hạn"]):
                    if any(k in names_str or k in cols_str for k in ["the_kho", "ton_kho", "phieu_nhap", "chi_tiet_nhap", "kho", "tồn", "lo_hang"]):
                        sc += 40
                # 4. Khách hàng
                if any(k in q_low for k in ["khách", "khach", "người mua", "ai mua"]):
                    if any(k in names_str or k in cols_str for k in ["khách", "khach"]):
                        sc += 30
                # 5. Khớp từ vựng câu hỏi
                for w in q_words:
                    if w in names_str:
                        sc += 15
                    elif w in cols_str:
                        sc += 5
                return sc

            scored_grps = sorted(items_to_render, key=_score_grp, reverse=True)
            # Giữ tối thiểu 3 nhóm, tối đa 5 nhóm có điểm >= 20
            kept_grps = [it for it in scored_grps if _score_grp(it) >= 20]
            if len(kept_grps) < 3:
                kept_grps = scored_grps[:3]
            elif len(kept_grps) > 5:
                kept_grps = kept_grps[:5]

            kept_keys = {it[0] for it in kept_grps}
            items_to_render = [it for it in items_to_render if it[0] in kept_keys]
            for it in scored_grps:
                if it[0] not in kept_keys:
                    omitted_table_names.extend([g[0] for g in it[1]])

        for cols_key, grp in items_to_render:
            tbl_names = [g[0] for g in grp]
            rep_df = grp[0][1]
            render_key = (business_id, tuple(tbl_names), cols_key, max_sample_rows)
            if render_key in self._schema_render_cache:
                lines.extend(self._schema_render_cache[render_key])
                continue
            block_start = len(lines)

            col_defs = []
            for col in cols_key:
                c_type = _infer_col_type(col, rep_df[col])
                col_defs.append(f'    "{col}" {c_type}')

            if len(tbl_names) > 2:
                tbl_desc = f'"{tbl_names[0]}" .. "{tbl_names[-1]}" ({len(tbl_names)} bảng: {", ".join(tbl_names[:3])}...)'
            else:
                tbl_desc = ", ".join(f'"{t}"' for t in tbl_names)

            role = _infer_semantic_role(tbl_names[0], list(cols_key))
            lines.append(f"\n/* [Bảng: {tbl_desc}] [{role}] */")
            lines.append(f'CREATE TABLE "{tbl_names[0]}" (\n' + ",\n".join(col_defs) + "\n);")

            if not rep_df.empty and max_sample_rows > 0:
                sample_records = rep_df[list(cols_key)].head(max_sample_rows).to_dict(orient="records")
                for r in sample_records:
                    row_json = json.dumps(r, ensure_ascii=False, default=str)
                    if len(row_json) > 350:
                        row_json = row_json[:347] + "..."
                    lines.append(f"  -- Mẫu: {row_json}")
            self._schema_render_cache[render_key] = lines[block_start:]

        if omitted_table_names:
            omitted_desc = ", ".join(f'"{t}"' for t in omitted_table_names)
            lines.append(f"\n/* Các bảng phụ khác ({omitted_desc}) đã nạp sẵn trong kết nối `con`, có thể SELECT nếu cần. */")

        return "\n".join(lines)
