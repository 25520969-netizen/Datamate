# -*- coding: utf-8 -*-
"""
Module 1: Ingestion & Schema Standardization Engine.
- Đọc và chuẩn hóa dữ liệu từ `.xlsx` (1 sheet, nhiều sheet theo tháng, hoặc nhiều sheet đa bảng),
  `.csv`, và `.sqlite` trên toàn bộ 30 doanh nghiệp (`biz_101` .. `biz_130`).
- Chuẩn hóa toàn bộ tên bảng, tên cột và chuỗi về Unicode NFC (tránh lỗi NFD trên macOS).
- Tự động gộp các bảng phân mảnh (`_1`, `_2`, `_2025`, `_2026`).
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
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

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
    """Chuẩn hóa chuỗi sang Unicode NFC và gộp khoảng trắng ngang thừa."""
    if isinstance(text, str):
        s = unicodedata.normalize("NFC", text).strip()
        return re.sub(r"[ \t]+", " ", s)
    return text


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
CACHE_VERSION = "v11"


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
            if 1 <= m_i <= 12 and 1 <= d_i <= 31:
                t_part = m_d.group(4)
                if t_part and not t_part.startswith("00:00"):
                    return f"{y_i:04d}-{m_i:02d}-{d_i:02d} {t_part}"
                return f"{y_i:04d}-{m_i:02d}-{d_i:02d}"
    return s


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
            df[col] = df[col].map(normalize_date_value_iso)
        elif df[col].dtype == object:
            df[col] = df[col].map(lambda x: to_nfc(x) if isinstance(x, str) else x)
        if col_low in ORDER_STATUS_COL_NAMES and df[col].dtype == object:
            df[col] = df[col].map(
                lambda x: CANONICAL_STATUS_MAP.get(x.lower().strip(), x) if isinstance(x, str) else x
            )
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

    # Loại bỏ các dòng bị lặp lại tiêu đề cột (như lỗi lặp header phân trang ở dev_010)
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

    if order_col is not None:
        # Thay thế chuỗi rỗng bằng NaN trên cột mã đơn để fill-down chính xác
        if df[order_col].dtype == object:
            df[order_col] = df[order_col].mask(df[order_col].astype(str).str.strip().isin(["", "nan", "<na>"]), pd.NA)

        # Phát hiện các dòng tổng kết / subtotal (ví dụ: Cộng ngày, Tổng tháng, Tổng cộng, Grand Total)
        # Các dòng này có mã đơn là NA và chứa từ khóa tổng kết ở một trong các cột văn bản.
        # Tuyệt đối KHÔNG fill-down mã đơn hoặc thông tin đơn hàng vào các dòng tổng kết này.
        subtotal_pat = re.compile(r"^\s*(?:cộng\s+ngày|tổng\s+tháng|tổng\s+cộng|grand\s+total|subtotal|cộng\b|tổng\b)", re.IGNORECASE)
        is_subtotal = pd.Series(False, index=df.index)
        na_order_mask = df[order_col].isna()
        if na_order_mask.any():
            text_cols = [c for c in df.columns if c != order_col and df[c].dtype == object]
            for tc in text_cols:
                is_subtotal = is_subtotal | (na_order_mask & df[tc].astype(str).str.contains(subtotal_pat, na=False))

        # Chỉ fill-down mã đơn cho các dòng chi tiết đơn hàng thông thường (không phải subtotal)
        # Dòng subtotal giữ nguyên mã đơn là NA để không bị gộp chung vào đơn hàng cuối cùng
        if df[order_col].isna().any() and df[order_col].notna().any():
            s_order = df[order_col].copy()
            s_order_filled = s_order.where(~is_subtotal).ffill()
            df[order_col] = s_order_filled.where(~is_subtotal, pd.NA)

        # Gán trạng thái 'Subtotal' cho các dòng subtotal nếu bảng có cột trạng thái
        if is_subtotal.any():
            for status_c in df.columns:
                if status_c.lower() in ORDER_STATUS_COL_NAMES or any(status_c.lower().startswith(p) for p in ["trạng thái", "trang thai", "tt đơn"]):
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
            for c in order_level_cols:
                if df[c].dtype == object:
                    df[c] = df[c].mask(df[c].astype(str).str.strip().isin(["", "nan", "<na>"]), pd.NA)
                if df[c].isna().any() and df[c].notna().any():
                    src_order = df[order_col].where(df[c].notna() & ~is_subtotal).ffill()
                    df[c] = df[c].where(~is_subtotal).ffill().where((src_order == df[order_col]) & ~is_subtotal, df[c])

            # Loại bỏ hai dòng của cùng một đơn giống nhau ở mọi ô (chỉ áp dụng cho các dòng thuộc về một đơn hàng)
            data_cols = [c for c in df.columns if not c.startswith("_")]
            order_rows = df[df[order_col].notna()].drop_duplicates(subset=data_cols, keep="first")
            no_order_rows = df[df[order_col].isna()]
            df = pd.concat([order_rows, no_order_rows]).sort_index().reset_index(drop=True)

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
            self.cache_root = self.data_root.parent.parent / ".cache" / "ingested"
        self._mem_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._readme_cache: Dict[str, str] = {}

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

    def get_shop_date_range(self, business_id: str) -> Tuple[str, str]:
        """Xác định ngày nhỏ nhất và lớn nhất có dữ liệu bán/nhập hàng của cửa hàng."""
        tables = self.load_tables(business_id)
        all_dts: List[str] = []
        for tname, df in tables.items():
            # Bỏ qua bảng thông tin tĩnh khách hàng, sản phẩm, bảng giá (chứa ngày sinh, ngày tạo sản phẩm không phải kỳ kinh doanh)
            if any(ex in tname.lower() for ex in ["khachhang", "khach_hang", "sanpham", "san_pham", "danhmuc", "danh_muc", "hanghoa", "hang_hoa", "banggia", "bang_gia"]):
                continue
            for c in df.columns:
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
        return ("2020-01-01", "2030-01-01")

    def load_tables(self, business_id: str, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Nạp toàn bộ bảng dữ liệu của một `business_id` thành dict[table_name, DataFrame].
        Tự động sử dụng Parquet cache nếu đã được chuẩn hóa trước đó.
        """
        if business_id in self._mem_cache:
            return self._mem_cache[business_id]

        biz_cache_dir = self.cache_root / business_id
        manifest_path = biz_cache_dir / "_manifest.json"

        if use_cache and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("version") == CACHE_VERSION:
                    tables: Dict[str, pd.DataFrame] = {}
                    for tbl_name, pq_file in manifest.get("tables", {}).items():
                        pq_path = biz_cache_dir / pq_file
                        if pq_path.exists():
                            tables[to_nfc(tbl_name)] = pd.read_parquet(pq_path)
                    if tables:
                        tables = self._enrich_table_synonyms_and_id(tables, business_id)
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
                        {"version": CACHE_VERSION, "business_id": business_id, "tables": manifest_tables},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

        tables = self._enrich_table_synonyms_and_id(tables, business_id)
        self._mem_cache[business_id] = tables
        return tables

    @staticmethod
    def _enrich_table_synonyms_and_id(tables: Dict[str, pd.DataFrame], business_id: str) -> Dict[str, pd.DataFrame]:
        """Bổ sung cột business_id và các alias cột phổ biến cho mọi bảng để chống Binder Error trong DuckDB/Pandas."""
        for tbl_name, df in tables.items():
            if "business_id" not in df.columns:
                df["business_id"] = business_id

            # Khách hàng
            if "Tên KH" in df.columns and "Tên khách" not in df.columns:
                df["Tên khách"] = df["Tên KH"]
            if "Tên KH" in df.columns and "Tên khách hàng" not in df.columns:
                df["Tên khách hàng"] = df["Tên KH"]
            if "Khách hàng" in df.columns and "Tên khách" not in df.columns:
                df["Tên khách"] = df["Khách hàng"]
            if "Mã KH" in df.columns and "Mã khách hàng" not in df.columns:
                df["Mã khách hàng"] = df["Mã KH"]
            if "Mã khách hàng" in df.columns and "Mã KH" not in df.columns:
                df["Mã KH"] = df["Mã khách hàng"]

            # Đơn hàng / Chứng từ
            for ord_col in ["Mã chứng từ", "Mã hóa đơn", "Số chứng từ", "SoCT", "Số HĐ", "Mã GD", "Mã đơn"]:
                if ord_col in df.columns and "Mã đơn hàng" not in df.columns:
                    df["Mã đơn hàng"] = df[ord_col]
                    break
            if "Mã đơn hàng" in df.columns and "Mã hóa đơn" not in df.columns:
                df["Mã hóa đơn"] = df["Mã đơn hàng"]

            # Trạng thái
            if "status_norm" in df.columns and "Trạng thái" not in df.columns:
                df["Trạng thái"] = df["status_norm"]
            if "Trạng thái" in df.columns and "status_norm" not in df.columns:
                df["status_norm"] = df["Trạng thái"]

            # Thời gian / Ngày
            if "Thời gian" in df.columns and "Ngày" not in df.columns:
                df["Ngày"] = df["Thời gian"]
            if "Ngày" in df.columns and "Thời gian" not in df.columns:
                df["Thời gian"] = df["Ngày"]

            # Sản phẩm
            if "Tên hàng" in df.columns and "Tên sản phẩm" not in df.columns:
                df["Tên sản phẩm"] = df["Tên hàng"]
            if "Tên sản phẩm" in df.columns and "Tên hàng" not in df.columns:
                df["Tên hàng"] = df["Tên sản phẩm"]
            if "Mã hàng" in df.columns and "Mã sản phẩm" not in df.columns:
                df["Mã sản phẩm"] = df["Mã hàng"]
            if "Mã sản phẩm" in df.columns and "Mã hàng" not in df.columns:
                df["Mã hàng"] = df["Mã sản phẩm"]

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

    def _ingest_from_raw(self, business_id: str) -> Dict[str, pd.DataFrame]:
        biz_dir = self.get_biz_dir(business_id)
        if not biz_dir.exists():
            return {}

        raw_tables: Dict[str, pd.DataFrame] = {}

        # Nếu thư mục là preprocessed có thư mục con tables/ (chứa các file CSV đã làm sạch 100%)
        tables_sub = biz_dir / "tables"
        if tables_sub.exists() and tables_sub.is_dir():
            for file_path in sorted(tables_sub.glob("*.csv")):
                stem = to_nfc(file_path.stem).strip()
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
                df = normalize_dataframe_nfc(df)
                raw_tables[stem] = df
            raw_tables = self._standardize_business_tables(raw_tables)
            return raw_tables

        for file_path in sorted(biz_dir.iterdir()):
            if file_path.name.startswith(".") or file_path.name.lower() == "readme.md":
                continue

            suffix = file_path.suffix.lower()
            stem = to_nfc(file_path.stem).strip()

            if suffix == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
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

        # Gộp các bảng phân mảnh có hậu tố _1, _2, _2025, _2026 (ví dụ SoBanHang_1 + SoBanHang_2026 -> SoBanHang)
        merged_groups: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
        for tbl_name, df in list(raw_tables.items()):
            m = re.match(r"^(.+?)_(\d+|202[4-9])$", tbl_name)
            if m:
                base_name = to_nfc(m.group(1)).strip()
                merged_groups.setdefault(base_name, []).append((tbl_name, df))

        for base_name, parts in merged_groups.items():
            if len(parts) >= 2 and base_name not in raw_tables:
                part_dfs = [p[1] for p in parts]
                combined = pd.concat(part_dfs, ignore_index=True)
                combined = apply_rulebook_section8_cleaning(combined)
                raw_tables[base_name] = combined

        # Tự động gộp các bảng bán hàng chia theo tháng thành 1 bảng `BanHang` duy nhất để truy vấn dễ dàng và siêu nhanh
        monthly_sales: List[pd.DataFrame] = []
        for tbl_name in list(raw_tables.keys()):
            if re.match(r"^(BanHang_T\d+|BanHang_Thang\d+|T\d+-\d{4}|BanHang_\d{4})", tbl_name, re.I):
                monthly_sales.append(raw_tables[tbl_name])
        if len(monthly_sales) >= 2 and "BanHang" not in raw_tables:
            first_cols = tuple(monthly_sales[0].columns)
            if all(tuple(d.columns) == first_cols for d in monthly_sales):
                comb = pd.concat(monthly_sales, ignore_index=True)
                raw_tables["BanHang"] = apply_rulebook_section8_cleaning(comb)

        raw_tables = self._standardize_business_tables(raw_tables)
        return raw_tables

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
           cho mọi dòng/bảng của đơn có trả hàng (kể cả trả 1 phần: Số lượng < 0, Số lượng trả > 0, bảng tra_hang).
        4. Bổ sung (denormalize) các cột tra cứu phổ biến (Thời gian, Trạng thái, Tên hàng, Hoạt chất, Thuốc kê đơn, Thành tiền)
           vào bảng chi tiết để tránh lỗi KeyError khi truy vấn trên 1 bảng.
        """
        # 1. Thu thập tên mặt hàng chuẩn từ các bảng danh mục / tồn kho (nếu có)
        catalog_name_map: Dict[str, str] = {}
        catalog_table_names = (
            "DanhSachSanPham",
            "Danh sách hàng hóa",
            "DanhMuc",
            "BanHang_HangHoa",
            "san_pham",
            "BaoCaoTonKho",
            "TonKho",
            "ton_kho",
        )
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
        for cat_tbl in catalog_table_names:
            if cat_tbl in tables:
                df_cat = tables[cat_tbl]
                for pcol in prod_col_candidates:
                    if pcol in df_cat.columns and df_cat[pcol].dtype == object:
                        for v in df_cat[pcol].dropna().astype(str):
                            v_clean = to_nfc(v)
                            if v_clean:
                                k = _accent_strip_key(v_clean)
                                prev = catalog_name_map.get(k)
                                if prev is None or (
                                    any(ord(ch) > 127 for ch in v_clean) and not any(ord(ch) > 127 for ch in prev)
                                ):
                                    catalog_name_map[k] = v_clean

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
                    by_key: Dict[str, List[str]] = {}
                    for v in non_null_vals:
                        by_key.setdefault(_accent_strip_key(v), []).append(v)
                    col_canon_map: Dict[str, str] = {}
                    for k, vs in by_key.items():
                        if k in catalog_name_map:
                            col_canon_map[k] = catalog_name_map[k]
                        elif len(set(vs)) > 1:
                            col_canon_map[k] = cls._choose_best_canonical_spelling(vs)
                    if col_canon_map:
                        df[pcol] = df[pcol].map(
                            lambda x: col_canon_map.get(_accent_strip_key(x), x) if isinstance(x, str) else x
                        )

            # Chuẩn hóa cột tên khách hàng
            has_cust_id = any(cid in df.columns for cid in cust_id_candidates)
            for ccol in cust_col_candidates:
                if ccol in df.columns and df[ccol].dtype == object:
                    non_null_vals = [str(x) for x in df[ccol].dropna().tolist() if str(x).strip()]
                    if not non_null_vals:
                        continue
                    by_key_c: Dict[str, List[str]] = {}
                    for v in non_null_vals:
                        by_key_c.setdefault(_accent_strip_key(v), []).append(v)
                    cust_canon_map: Dict[str, str] = {}
                    for k, vs in by_key_c.items():
                        if k in walkin_keys:
                            cust_canon_map[k] = "Khách lẻ"
                        elif not has_cust_id and len(set(vs)) > 1:
                            cust_canon_map[k] = cls._choose_best_canonical_spelling(vs)
                    if cust_canon_map:
                        df[ccol] = df[ccol].map(
                            lambda x: cust_canon_map.get(_accent_strip_key(x), x) if isinstance(x, str) else x
                        )

        # 2. Thu thập danh sách mã đơn có trả hàng (từ bảng lô, bảng trả hàng, dòng số lượng âm, hoặc dòng trạng thái Trả hàng)
        returned_order_ids: set = set()
        all_order_id_cols = set(ORDER_ID_CANDIDATES) | {"Order ID", "MaDon", "ma_don"}

        if "tra_hang" in tables and "MaDon" in tables["tra_hang"].columns:
            returned_order_ids.update(tables["tra_hang"]["MaDon"].dropna().astype(str).tolist())

        for tbl_name, df in tables.items():
            ord_col = next((c for c in df.columns if c in all_order_id_cols), None)
            if not ord_col:
                continue
            # Kiểm tra cột số lượng trả > 0 (trong DanhSachHoaDonChiTietLo, DonHang_Lo, BanHang_BanTheoLo)
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

            # Nếu cột 'Thời gian' chỉ chứa ngày (không có giờ phút), nhưng có cột datetime chi tiết hơn, ưu tiên cột chi tiết
            if "Thời gian" in df.columns:
                detailed_dt = next(
                    (c for c in df.columns if c != "Thời gian" and any(k in c.lower() for k in ["thời gian", "thoi gian", "created", "datetime", "giờ", "gio"]) and df[c].astype(str).str.contains(r"\d{1,2}:\d{2}").any()),
                    None
                )
                if detailed_dt:
                    df["Thời gian"] = df[detailed_dt]

            # Tạo alias cột trạng thái chuẩn ('Trạng thái') trên mọi bảng có cột trạng thái
            st_found = next(
                (c for c in df.columns if c.lower().strip() in ORDER_STATUS_COL_NAMES or any(c.lower().strip().startswith(p) for p in ["trạng thái", "trang thai", "tt đơn", "tình trạng", "ghi chú tt"])),
                None
            )
            if st_found and "Trạng thái" not in df.columns:
                df["Trạng thái"] = df[st_found]

            if "Trạng thái" in df.columns and "status_norm" not in df.columns:
                def _norm_status_val(v: Any) -> str:
                    s = str(v).lower().strip()
                    if any(k in s for k in ["hủy", "huỷ", "cancel"]):
                        return "Đã hủy"
                    if any(k in s for k in ["trả", "tra hang", "return"]):
                        return "Trả hàng"
                    if any(k in s for k in ["hoàn thành", "thành công", "xong", "đã lấy", "completed", "finish"]):
                        return "Hoàn thành"
                    return str(v)
                df["status_norm"] = df["Trạng thái"].map(_norm_status_val)

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
                        "tien hang", "tiền hàng", "thành tiền", "thanh tien", "thành tiền (đ)", "amount", "tổng tiền hàng"
                    ])),
                    None
                )
                if pure_rev_col:
                    df["revenue_norm"] = df[pure_rev_col]
                else:
                    tot_rev_col = next(
                        (c for c in df.columns if any(c.lower().strip() == p for p in [
                            "tổng tiền", "tongtien", "tongcong", "tổng cộng", "tiền", "thành tiền"
                        ])),
                        None
                    )
                    if tot_rev_col:
                        df["revenue_norm"] = df[tot_rev_col]

            ord_col = next((c for c in df.columns if c in all_order_id_cols), None)
            if not ord_col:
                continue
            # Nếu bảng có nhiều dòng trên cùng 1 mã đơn và có cột ngày đặt + cột trạng thái đơn (như SoBanHang ở biz_105, biz_117)
            st_col = next((c for c in df.columns if c.lower().strip() in ORDER_STATUS_COL_NAMES), None)
            dt_col = next((c for c in df.columns if c in order_date_col_candidates), None)
            if dt_col and st_col and not df[ord_col].is_unique and df[dt_col].dtype == object:
                min_dates = df.groupby(ord_col, sort=False)[dt_col].min()
                df[dt_col] = df[ord_col].map(min_dates).fillna(df[dt_col])
            if st_col and returned_order_ids:
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
                def _norm_st(v: Any) -> str:
                    s = str(v).lower().strip()
                    if any(k in s for k in ["hủy", "huỷ", "cancel"]):
                        return "Đã hủy"
                    if any(k in s for k in ["trả", "tra hang", "return"]):
                        return "Trả hàng"
                    if any(k in s for k in ["hoàn thành", "thành công", "xong", "đã lấy", "completed", "finish", "hoàn tất", "hoan tat"]):
                        return "Hoàn thành"
                    return str(v)
                df["status_norm"] = df["Trạng thái"].map(_norm_st)
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
                df = raw_df.iloc[best_row_idx + 1 :].copy()
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
        - Nếu các sheet có cấu trúc cột khác nhau (như DuLieuBanHang.xlsx, Bao_cao_ban_hang.xlsx, DuLieuXuat.xlsx)
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

        # Kiểm tra xem tất cả các sheet có cùng tập hợp cột hay không
        first_cols = tuple(sheet_dfs[0][1].columns)
        all_same_schema = all(tuple(df.columns) == first_cols for _, df in sheet_dfs)

        if all_same_schema:
            annotated_dfs: List[pd.DataFrame] = []
            for s_name, df in sheet_dfs:
                df_copy = df.copy()
                df_copy["_sheet_name"] = s_name
                annotated_dfs.append(df_copy)
            merged_df = pd.concat(annotated_dfs, ignore_index=True)
            merged_df = apply_rulebook_section8_cleaning(merged_df)
            return {file_stem: merged_df}

        # Các sheet là các bảng nghiệp vụ khác nhau
        out: Dict[str, pd.DataFrame] = {}
        for s_name, df in sheet_dfs:
            out[s_name] = apply_rulebook_section8_cleaning(df)
        return out

    def create_duckdb_connection(self, business_id: str) -> Tuple[duckdb.DuckDBPyConnection, Dict[str, pd.DataFrame]]:
        """Tạo kết nối DuckDB in-memory đã đăng ký sẵn toàn bộ các bảng của `business_id`."""
        tables = self.load_tables(business_id)
        tables = self._enrich_table_synonyms_and_id(tables, business_id)
        con = duckdb.connect(database=":memory:")
        for tbl_name, df in tables.items():
            con.register(tbl_name, df)
        return con, tables

    def build_compact_schema_prompt(self, business_id: str, max_sample_rows: int = 0, question: str = "") -> str:
        """
        Tạo Compact Schema Metadata + nội dung README.md của shop để đưa vào Prompt của LLM.
        Đảm bảo tổng độ dài gọn gàng (nằm gọn trong giới hạn 8192 token của vLLM cả khi retry).
        """
        readme_text = self.get_readme(business_id).strip()
        # Nếu README có danh sách quy đổi đơn vị, chỉ giữ các dòng khớp từ khóa câu hỏi + 3 dòng mẫu
        readme_lines = readme_text.splitlines()
        if question and any("**Quy đổi đơn vị**" in ln for ln in readme_lines):
            q_words = [w for w in re.split(r"\W+", question.lower()) if len(w) >= 2]
            filtered_readme: List[str] = []
            in_conv = False
            conv_kept = 0
            for ln in readme_lines:
                if "**Quy đổi đơn vị**" in ln:
                    in_conv = True
                    filtered_readme.append(ln)
                    continue
                if in_conv and ln.strip().startswith("- ") and not ln.strip().startswith("- **"):
                    ln_low = ln.lower()
                    if any(w in ln_low for w in q_words) or conv_kept < 2:
                        filtered_readme.append(ln)
                        conv_kept += 1
                else:
                    if in_conv and ln.strip().startswith("- **"):
                        in_conv = False
                    filtered_readme.append(ln)
            readme_text = "\n".join(filtered_readme)

        tables = self.load_tables(business_id)

        # Nếu đã có bảng gộp (ví dụ SoBanHang từ SoBanHang_1 + SoBanHang_2026, hoặc BanHang gộp từ các tháng), ẩn các bảng mảnh trong prompt
        part_names_to_hide = set()
        for tbl_name in tables:
            m = re.match(r"^(.+?)_(\d+|202[4-9])$", tbl_name)
            if m and m.group(1).strip() in tables:
                part_names_to_hide.add(tbl_name)
        if "BanHang" in tables:
            for tbl_name in tables:
                if re.match(r"^(BanHang_T\d+|BanHang_Thang\d+|T\d+-\d{4}|BanHang_\d{4})", tbl_name, re.I):
                    part_names_to_hide.add(tbl_name)

        if len(readme_text) > 2400:
            readme_text = readme_text[:2400] + "\n...(xem thêm trong tài liệu shop)"

        lines = [
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
            c_low = col_name.lower()
            if any(k in c_low for k in ["ngày", "ngay", "thời gian", "date", "hạn dùng", "hsd"]):
                return "TIMESTAMP"
            if any(k in c_low for k in ["tiền", "tien", "giá", "gia", "doanh thu", "dthu", "lãi", "nợ", "chiết khấu"]):
                return "DOUBLE"
            if any(k in c_low for k in ["số lượng", "so luong", "qty", "tồn", "ton"]):
                return "DOUBLE"
            if series.dtype in ["float64", "float32"]:
                return "DOUBLE"
            if series.dtype in ["int64", "int32"]:
                return "BIGINT"
            non_null = series.dropna()
            if len(non_null) > 0 and series.dtype == object:
                uniq = non_null.unique().tolist()
                if 1 < len(uniq) <= 4 and all(len(str(u)) <= 15 for u in uniq):
                    return f"VARCHAR {uniq}"
            return "VARCHAR"

        # Nhóm các bảng có cùng cấu trúc cột để tránh lặp lại hàng chục lần (như các bảng tháng của dev_005, dev_014)
        by_cols: Dict[Tuple[str, ...], List[Tuple[str, pd.DataFrame]]] = {}
        for tbl_name, df in tables.items():
            if tbl_name in part_names_to_hide:
                continue
            cols_key = tuple(c for c in df.columns if c != "_sheet_name")
            by_cols.setdefault(cols_key, []).append((tbl_name, df))

        for cols_key, grp in by_cols.items():
            tbl_names = [g[0] for g in grp]
            rep_df = grp[0][1]

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
                sample_records = rep_df.head(max_sample_rows).to_dict(orient="records")
                for r in sample_records:
                    row_json = json.dumps(r, ensure_ascii=False, default=str)
                    if len(row_json) > 350:
                        row_json = row_json[:347] + "..."
                    lines.append(f"  -- Mẫu: {row_json}")

        return "\n".join(lines)
