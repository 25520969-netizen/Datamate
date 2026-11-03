# -*- coding: utf-8 -*-
"""
Module 5: Post-Processing & Penalized Accuracy Guard (Abstention Policy).
- Chuẩn hóa đầu ra khớp 100% với quy chuẩn chấm thi tại Mục 9 Rulebook:
  * number: JSON number; tiền VND hoặc trung bình khoản tiền làm tròn về đồng (nửa đồng làm tròn lên);
    số trung bình khác làm tròn 2 chữ số thập phân.
  * percent: JSON number làm tròn 2 chữ số thập phân, không kèm ký hiệu '%'.
  * date: 'YYYY-MM-DD' hoặc 'YYYY-MM'.
  * boolean: true / false.
  * string: Unicode NFC, không rỗng.
  * list: mảng JSON có thứ tự.
  * table: {"columns": [...], "rows": [[...], ...]}.
  * unanswerable: {"type": "unanswerable", "reason": "..."}.
- Chiến lược phòng thủ điểm số (Penalized Accuracy Guard):
  * Sai bị phạt -1 điểm, bỏ qua ('abstain') được 0 điểm.
  * Tự động kích hoạt `{"type": "abstain"}` khi kết quả là None, NaN, rỗng, lỗi thực thi,
    hoặc vi phạm kiểm tra tính hợp lý (sanity check).
"""

import math
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


def strip_accents(text: Any) -> str:
    """Chuẩn hóa chuỗi tiếng Việt: NFC, đổi đ/Đ thành d, bỏ dấu tổ hợp, chữ thường."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip()
    s = s.replace("đ", "d").replace("Đ", "d")
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


MONEY_KEYWORDS = (
    "bao nhiêu tiền",
    "bao nhiu tien",
    "mấy đồng",
    "doanh thu",
    "dthu",
    "thành tiền",
    "giá trị",
    "tổng tiền",
    "tiền hàng",
    "tiền bán",
    "tiền nhập",
    "tiền vốn",
    "giá vốn",
    "giá bán",
    "đơn giá",
    "giá nhập",
    "giá bình quân",
    "giá trung bình",
    "giá",
    "chi phí",
    "lãi gộp",
    "lãi ròng",
    "còn nợ",
    "nợ bao nhiêu",
    "thu nợ",
    "trả nợ",
    "nợ thêm",
    "tiền",
    "đồng",
    "vnđ",
    "chi bao nhiêu",
    "chi tiêu",
    "lãi",
    "tiền lời",
    "thu về",
    "giá trị đơn",
)

NON_NEGATIVE_KEYWORDS = (
    "tổng doanh thu",
    "bán được bao nhiêu",
    "có bao nhiêu đơn",
    "đếm số đơn",
    "bao nhiêu khách",
    "bao nhiêu đợt",
    "tổng giá trị các phiếu nhập",
    "đơn giá bao nhiêu",
)

CHANGE_OR_DIFF_KEYWORDS = (
    "chênh lệch",
    "chenh lech",
    "tăng",
    "giảm",
    "hơn",
    "kém",
    "thay đổi",
    "lãi ròng",
    "lãi gộp",
    "nhập trừ bán",
)


def round_half_up_int(val: float) -> int:
    """Làm tròn về số nguyên gần nhất, nửa đồng (0.5) làm tròn lên theo Mục 3 & Mục 9 Rulebook."""
    d = Decimal(str(val))
    return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def round_half_up_2dec(val: float) -> float:
    """Làm tròn 2 chữ số thập phân chuẩn kế toán."""
    d = Decimal(str(val))
    rounded = float(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if rounded.is_integer():
        return int(rounded)
    return rounded


def is_money_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in MONEY_KEYWORDS)


def is_Strictly_non_negative_question(question: str) -> bool:
    q = question.lower()
    if any(k in q for k in CHANGE_OR_DIFF_KEYWORDS):
        return False
    return any(k in q for k in NON_NEGATIVE_KEYWORDS)


def normalize_date_str(val: Any) -> Optional[str]:
    if not isinstance(val, str):
        val = str(val)
    s = val.strip()
    # Chuẩn YYYY-MM-DD hoặc YYYY-MM
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", s):
        return s
    # Chuẩn YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DD
    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2})[T\s]", s)
    if m_iso:
        return m_iso.group(1)
    # Chuẩn DD/MM/YYYY -> YYYY-MM-DD
    m_dmy = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](20\d{2})$", s)
    if m_dmy:
        return f"{int(m_dmy.group(3)):04d}-{int(m_dmy.group(2)):02d}-{int(m_dmy.group(1)):02d}"
    # Chuẩn MM/YYYY -> YYYY-MM
    m_my = re.match(r"^(?:tháng\s*)?(\d{1,2})[/-](20\d{2})$", s.lower())
    if m_my:
        return f"{int(m_my.group(2)):04d}-{int(m_my.group(1)):02d}"
    return None


def strip_trailing_measurement_in_parens(s: str) -> str:
    """
    Chỉ loại bỏ ngoặc đơn nếu bên trong là số tiền, số lượng hoặc số đo do LLM tự ghi chú thêm
    (ví dụ: " (1.605.000)", " (307 ly)", " (15 chai)", " (50 đ)", " (256)", " (1507)").
    TUYỆT ĐỐI KHÔNG BỎ hậu tố size/liều lượng/quy cách là một phần của tên sản phẩm:
    (M), (L), (S), (XL), (chai 500ml), (hộp 24 gói), (500mg), v.v.
    """
    if not isinstance(s, str):
        return s
    m = re.search(r"\s*\(([^)]+)\)\s*$", s)
    if not m:
        return s.strip()
    inside = m.group(1).strip()
    # 1. Nếu bên trong là số thuần túy (ví dụ: (256), (1507), (193), (1.605.000)) -> chắc chắn là số đo/số lượng do LLM tự chú thích thêm -> LOẠI BỎ
    if re.match(r"^[\d\.,\s]+$", inside):
        return s[:m.start()].strip()
    # 2. Nếu chứa số tiền hoặc số lượng kèm đơn vị (1.000đ, 307 ly, 10 cái, 15 chai) -> LOẠI BỎ
    #    (DataMateAgent._ground_strings khôi phục lại tên gốc nếu "(24 lon)" thực ra là một phần tên trong dữ liệu)
    if re.search(r"[\d\.,\s]+(?:đ|vnđ|đồng|ly|chai|nồi|hộp|lon|cái|kg|g|%|bill|đơn|k|cuộn)", inside, re.I):
        return s[:m.start()].strip()
    # 3. Nếu là size ngắn thuần chữ cái (M, L, S, XL, XXL) hoặc quy cách sản phẩm (500mg, 100ml, v.v.), GIỮ NGUYÊN
    if re.match(r"^[a-zA-Z\s]{1,3}$", inside) or re.search(r"(?:mg|ml|kg|gam|g|hộp|vỉ|chai|gói)\s*\d*", inside, re.I):
        return s.strip()
    return s.strip()


clean_entity_label = strip_trailing_measurement_in_parens


def postprocess_and_guard(
    question_id: str,
    question: str,
    raw_output: Any,
    expected_type: str = "number",
    code: Optional[str] = None,
    reject_walk_in: bool = True,
) -> Dict[str, Any]:
    """
    Chuẩn hóa kết quả từ Sandbox và áp dụng cơ chế bảo vệ Penalized Accuracy.
    Nếu raw_output đã là một dict có cấu trúc `{"type": ..., "answer": ...}`, sử dụng trực tiếp `type` đó.
    Nếu phát hiện bất kỳ dấu hiệu lỗi, rỗng, NaN hoặc không hợp lý -> trả về `{"id": question_id, "type": "abstain"}`.
    """
    abstain_resp = {"id": question_id, "type": "abstain"}

    if raw_output is None:
        return abstain_resp

    ans_type = expected_type
    ans_val = raw_output
    reason_val: Optional[str] = None

    if isinstance(raw_output, dict):
        if "type" in raw_output:
            ans_type = str(raw_output.get("type", "")).strip().lower()
        if "answer" in raw_output:
            ans_val = raw_output["answer"]
        elif "columns" in raw_output and "rows" in raw_output:
            ans_val = {"columns": raw_output["columns"], "rows": raw_output["rows"]}
        elif "value" in raw_output:
            ans_val = raw_output["value"]
        elif "result" in raw_output:
            ans_val = raw_output["result"]
        if "reason" in raw_output and isinstance(raw_output["reason"], str):
            reason_val = raw_output["reason"]

    if ans_type == "abstain":
        return abstain_resp

    if ans_type == "unanswerable":
        return {
            "id": question_id,
            "type": "unanswerable",
            "reason": reason_val or "Dữ liệu phát hành không đủ để trả lời câu hỏi này.",
        }

    # Tự động suy luận lại type nếu raw_output mang kiểu dữ liệu rõ ràng khác với expected_type
    if hasattr(ans_val, "columns") and hasattr(ans_val, "values"):
        ans_type = "table"
        ans_val = {"columns": [str(c) for c in ans_val.columns], "rows": ans_val.values.tolist()}
    elif isinstance(ans_val, dict) and "columns" in ans_val and "rows" in ans_val:
        ans_type = "table"
    elif isinstance(ans_val, list) and len(ans_val) > 1 and ans_type not in ("list", "table"):
        ans_type = "list"
    elif isinstance(ans_val, bool):
        ans_type = "boolean"

    # Hiệu chỉnh kiểu khi code gán nhầm type nhưng giá trị chuyển được KHÔNG mơ hồ sang kiểu câu hỏi đòi
    # (sai type là sai dù đúng giá trị, Mục 9)
    if expected_type == "percent" and ans_type == "number":
        ans_type = "percent"
    elif expected_type == "date" and ans_type == "string" and isinstance(ans_val, str) and re.match(r"^\d{4}-\d{2}(-\d{2})?$", ans_val.strip()):
        ans_type = "date"
    elif expected_type == "boolean" and ans_type == "string" and isinstance(ans_val, str) and unicodedata.normalize("NFC", ans_val).strip().lower() in (
        "true", "false", "đúng", "sai", "có", "không"
    ):
        ans_type = "boolean"
    elif expected_type == "string" and ans_type == "list" and isinstance(ans_val, list) and len(ans_val) == 1 and isinstance(ans_val[0], str):
        ans_type = "string"

    # Tự động unpack triệt để các cấu trúc lồng nhau (list/tuple 1 phần tử từ DuckDB fetchall) về scalar
    while isinstance(ans_val, (list, tuple)) and len(ans_val) == 1 and ans_type in ("number", "percent", "string", "date", "boolean"):
        ans_val = ans_val[0]

    # 1. Xử lý NUMBER
    if ans_type == "number":
        q_low = question.lower()
        q_clean = strip_accents(question)
        if isinstance(ans_val, str):
            s_clean = ans_val.strip().replace("đ", "").replace("VND", "").replace("vnd", "").replace("d", "").strip()
            from src.ingest import parse_vn_numeric_value
            parsed_num = parse_vn_numeric_value(s_clean)
            if parsed_num is not None:
                ans_val = parsed_num
            else:
                try:
                    ans_val = float(s_clean.replace(",", ""))
                except ValueError:
                    pass

        if isinstance(ans_val, bool) or not isinstance(ans_val, (int, float)):
            return abstain_resp
        if math.isnan(ans_val) or math.isinf(ans_val):
            return abstain_resp

        if is_Strictly_non_negative_question(question) and float(ans_val) < 0:
            return abstain_resp

        # Sanity check: Các câu hỏi doanh thu, số lượng, số đơn, tiền nợ không bao giờ âm nếu không phải câu hỏi chênh lệch
        if float(ans_val) < 0 and not any(k in q_low for k in CHANGE_OR_DIFF_KEYWORDS):
            return abstain_resp

        if is_money_question(question) or float(ans_val).is_integer():
            final_num: Any = round_half_up_int(float(ans_val))
        else:
            final_num = round_half_up_2dec(float(ans_val))

        return {"id": question_id, "type": "number", "answer": final_num}

    # 2. Xử lý PERCENT
    if ans_type == "percent":
        q_low = question.lower()
        q_clean = strip_accents(question)
        if isinstance(ans_val, str):
            s_clean = ans_val.strip().rstrip("%").strip().replace(",", ".")
            try:
                ans_val = float(s_clean)
            except ValueError:
                return abstain_resp
        if isinstance(ans_val, bool) or not isinstance(ans_val, (int, float)):
            return abstain_resp
        if math.isnan(ans_val) or math.isinf(ans_val):
            return abstain_resp

        # Sanity check: tỉ lệ phần trăm không thể < -100% hoặc vượt quá 5000% (tăng trưởng doanh thu từ cơ sở nhỏ)
        if float(ans_val) < -100.0 or abs(float(ans_val)) > 5000.0:
            return abstain_resp

        # Quy đổi tỷ số thập phân (0.403 -> 40.3) chỉ khi chắc chắn code KHÔNG nhân 100.
        # Phần trăm thật nhỏ hơn 1% (vd tăng 0.45%) là hợp lệ, không được nhân thêm 100.
        # Không có code để kiểm tra (gọi trực tiếp) -> giữ hành vi cũ.
        if 0.0 < abs(float(ans_val)) < 1.0:
            multiplied_in_code = code is not None and bool(re.search(r"\b100(?:\.0*)?\b", code))
            if not multiplied_in_code:
                ans_val = float(ans_val) * 100.0

        final_pct = round_half_up_2dec(float(ans_val))

        # Sanity Guard: Tỉ trọng / Tỉ lệ đóng góp không thể vượt quá 100% hoặc nhỏ hơn 0%
        if any(k in q_clean for k in ["chiem", "ti trong", "ty trong", "dong gop", "ti le don huy", "ti le huy"]):
            if final_pct > 100.0 or final_pct < 0.0:
                return abstain_resp

        return {"id": question_id, "type": "percent", "answer": final_pct}

    # 3. Xử lý DATE
    if ans_type == "date":
        norm_dt = normalize_date_str(ans_val)
        if not norm_dt:
            return abstain_resp
        # Mục 9: câu hỏi theo tháng ("tháng nào", "tháng mấy") đáp án là "YYYY-MM"
        q_acc = strip_accents(question)
        if len(norm_dt) == 10 and re.search(r"\bthang\s*(?:nao|may)\b", q_acc) and not re.search(r"\bngay\s*(?:nao|may|bao nhieu)\b", q_acc):
            norm_dt = norm_dt[:7]
        return {"id": question_id, "type": "date", "answer": norm_dt}

    # 4. Xử lý BOOLEAN
    if ans_type == "boolean":
        if isinstance(ans_val, str):
            s_val = unicodedata.normalize("NFC", ans_val).strip().lower()
            if s_val in ("true", "đúng", "dung", "có", "co", "tăng", "tang", "cao hơn", "cao hon"):
                ans_val = True
            elif s_val in ("false", "sai", "không", "khong", "ko", "k", "giảm", "giam", "thấp hơn", "thap hon"):
                ans_val = False
        elif isinstance(ans_val, (int, float)) and not isinstance(ans_val, bool):
            if ans_val == 1:
                ans_val = True
            elif ans_val == 0:
                ans_val = False

        if not isinstance(ans_val, bool):
            return abstain_resp
        return {"id": question_id, "type": "boolean", "answer": ans_val}

    # 5. Xử lý STRING
    if ans_type == "string":
        if not isinstance(ans_val, str):
            return abstain_resp
        clean_str = unicodedata.normalize("NFC", ans_val).strip()

        # Nếu model trả về chuỗi ghép nhiều đối tượng bằng dấu phẩy mà câu hỏi hỏi danh sách/top N/nhiều đối tượng
        # -> Tự động chuyển đổi thành type "list" chuẩn JSON
        # Không dùng "các"/"những": "Trong các món, món nào..." hỏi một đối tượng, và tên món có thể chứa dấu phẩy
        is_list_q = (
            expected_type == "list"
            or any(kw in question.lower() for kw in ["top", "danh sách", "liệt kê", "liet ke"])
            or bool(re.search(r"\b(?<!\d)(?:[2-9]|10)\s+(?:khách|khach|mặt hàng|mat hang|món|mon|sản phẩm|san pham|thợ|tho|nhân viên|nhan vien|thuốc|thuoc)\b", question.lower()))
        )
        if "," in clean_str and is_list_q:
            parts = [
                strip_trailing_measurement_in_parens(unicodedata.normalize("NFC", p)).strip()
                for p in clean_str.split(",")
                if p.strip()
            ]
            parts = [p for p in parts if not (reject_walk_in and p.lower() in WALK_IN_LABELS)]
            if len(parts) > 1:
                return {"id": question_id, "type": "list", "answer": parts}

        # Loại bỏ các số đo / số tiền trong ngoặc đơn (vd: 'KH0014 (1,605,000)' -> 'KH0014', 'Trà đào (307 ly)' -> 'Trà đào')
        # Nhưng TUYỆT ĐỐI giữ nguyên hậu tố size như '(M)', '(L)', '(500mg)'
        clean_str = strip_trailing_measurement_in_parens(clean_str)

        # Chỉ khi rulebook ghi khách lẻ không phải khách (không xếp hạng)
        if reject_walk_in and clean_str.lower() in WALK_IN_LABELS:
            return abstain_resp

        if not clean_str or clean_str.lower() in ("none", "nan", "null", "không có", "không có mặt hàng nào"):
            return abstain_resp
        return {"id": question_id, "type": "string", "answer": clean_str}

    # 6. Xử lý LIST
    if ans_type == "list":
        if isinstance(ans_val, str):
            clean_s = unicodedata.normalize("NFC", ans_val).strip()
            if clean_s.startswith("[") and clean_s.endswith("]"):
                try:
                    import ast
                    parsed_l = ast.literal_eval(clean_s)
                    if isinstance(parsed_l, list):
                        ans_val = parsed_l
                except Exception:
                    pass
            if isinstance(ans_val, str):
                if "\n" in clean_s:
                    parts = clean_s.split("\n")
                elif "," in clean_s:
                    parts = clean_s.split(",")
                else:
                    parts = [clean_s]
                ans_val = [
                    strip_trailing_measurement_in_parens(re.sub(r"^\s*[-*•\d\.\)]+\s*", "", p))
                    for p in parts
                ]
                ans_val = [p for p in ans_val if p and not (reject_walk_in and p.lower() in WALK_IN_LABELS)]

        if not isinstance(ans_val, list) or len(ans_val) == 0:
            return abstain_resp

        # Tự động làm phẳng (flatten) các danh sách lồng nhau / kết quả fetchall() từ DuckDB
        # Ví dụ: [('Panadol...',), ('Smecta...',)] hoặc [['A'], ['B']] hoặc 1 hàng duy nhất: [[val1, val2]]
        if len(ans_val) == 1 and isinstance(ans_val[0], (list, tuple)):
            ans_val = list(ans_val[0])
        elif all(isinstance(x, (list, tuple)) and len(x) == 1 for x in ans_val):
            ans_val = [x[0] for x in ans_val]
        elif any(isinstance(x, (list, tuple)) for x in ans_val):
            flattened = []
            for x in ans_val:
                if isinstance(x, (list, tuple)):
                    flattened.extend(x)
                else:
                    flattened.append(x)
            ans_val = flattened

        cleaned_list: List[Any] = []
        money_flag = is_money_question(question)
        for item in ans_val:
            if item is None:
                continue
            if isinstance(item, bool):
                cleaned_list.append(item)
            elif isinstance(item, (int, float)):
                if math.isnan(item) or math.isinf(item):
                    continue
                if money_flag or float(item).is_integer():
                    cleaned_list.append(round_half_up_int(float(item)))
                else:
                    cleaned_list.append(round_half_up_2dec(float(item)))
            elif isinstance(item, str):
                s_item = unicodedata.normalize("NFC", item).strip()
                # Loại bỏ số tiền/số lượng trong ngoặc đơn của item (giữ nguyên size như (M), (L))
                s_item = strip_trailing_measurement_in_parens(s_item)
                # Khử khách lẻ
                if s_item.lower() in ("none", "nan", "null") or (reject_walk_in and s_item.lower() in WALK_IN_LABELS):
                    continue
                # Tránh chuỗi rác dài bất thường
                if len(s_item) > 120:
                    return abstain_resp
                if s_item:
                    cleaned_list.append(s_item)
            else:
                return abstain_resp

        # Khử trùng lặp tên giữ nguyên thứ tự. Số KHÔNG khử trùng: [3, 3] (3 thùng 3 chai) là hai giá trị khác nhau.
        seen = set()
        deduped = []
        for x in cleaned_list:
            if isinstance(x, str):
                if x in seen:
                    continue
                seen.add(x)
            deduped.append(x)

        if not deduped:
            return abstain_resp
        return {"id": question_id, "type": "list", "answer": deduped}

    # 7. Xử lý TABLE
    if ans_type == "table":
        # Chuẩn hóa nếu model trả về danh sách hàng [[...], ...] hoặc list of dicts [{'col1': val1, ...}] (chuẩn Pandas orient='records')
        if isinstance(ans_val, list):
            if len(ans_val) > 0 and isinstance(ans_val[0], (list, tuple)):
                # Kiểm tra nếu dòng 0 là tiêu đề cột
                if all(isinstance(c, str) for c in ans_val[0]) and len(ans_val) > 1 and not all(isinstance(c, str) for c in ans_val[1]):
                    cols = [str(c) for c in ans_val[0]]
                    rows = ans_val[1:]
                else:
                    cols = [f"col_{i+1}" for i in range(len(ans_val[0]))]
                    rows = ans_val
                ans_val = {"columns": cols, "rows": rows}
            elif len(ans_val) > 0 and isinstance(ans_val[0], dict):
                cols = list(ans_val[0].keys())
                rows = [[r.get(c) for c in cols] for r in ans_val]
                ans_val = {"columns": cols, "rows": rows}
            else:
                return abstain_resp
        elif isinstance(ans_val, dict):
            if "rows" in ans_val and "columns" not in ans_val:
                rows_cand = ans_val.get("rows")
                if isinstance(rows_cand, list) and len(rows_cand) > 0 and isinstance(rows_cand[0], (list, tuple)):
                    ans_val["columns"] = [f"col_{i+1}" for i in range(len(rows_cand[0]))]
                else:
                    return abstain_resp
        else:
            return abstain_resp

        cols = ans_val.get("columns")
        rows = ans_val.get("rows")
        if not isinstance(cols, list) or not isinstance(rows, list) or len(cols) == 0 or len(rows) == 0:
            return abstain_resp

        clean_cols = [unicodedata.normalize("NFC", str(c)).strip() for c in cols]
        clean_rows: List[List[Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != len(clean_cols):
                return abstain_resp
            c_row: List[Any] = []
            is_walkin_row = False
            for idx, cell in enumerate(row):
                if cell is None:
                    return abstain_resp
                if idx == 0 and isinstance(cell, str):
                    s_cell = strip_trailing_measurement_in_parens(unicodedata.normalize("NFC", cell)).strip()
                    if not s_cell:
                        return abstain_resp
                    # Chỉ khi rulebook ghi khách lẻ không phải khách: không có dòng trong bảng theo khách
                    if reject_walk_in and s_cell.lower() in WALK_IN_LABELS:
                        is_walkin_row = True
                        break
                    c_row.append(s_cell)
                elif isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    if math.isnan(cell) or math.isinf(cell):
                        return abstain_resp
                    if float(cell).is_integer():
                        c_row.append(round_half_up_int(float(cell)))
                    else:
                        c_row.append(round_half_up_2dec(float(cell)))
                elif isinstance(cell, str):
                    s_str = strip_trailing_measurement_in_parens(unicodedata.normalize("NFC", cell)).strip()
                    c_row.append(s_str)
                else:
                    return abstain_resp
            if not is_walkin_row:
                clean_rows.append(c_row)

        if not clean_rows:
            return abstain_resp

        return {
            "id": question_id,
            "type": "table",
            "answer": {"columns": clean_cols, "rows": clean_rows},
        }

    return abstain_resp


WALK_IN_LABELS = ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai", "khach vang lai")


def _scorer_norm(text: Any) -> str:
    """Chuẩn hoá giống luật chấm chuỗi: NFC, chữ thường, ký tự không phải chữ/số -> khoảng trắng. GIỮ dấu."""
    nfc = unicodedata.normalize("NFC", str(text)).lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in nfc)
    return re.sub(r"\s+", " ", cleaned).strip()


def are_strings_in_consensus(a1: Any, a2: Any) -> bool:
    """
    Hai tên đồng thuận khi trùng nhau sau chuẩn hoá của luật chấm (hoa/thường, khoảng trắng, dấu câu),
    sau khi bỏ chú thích số đo LLM tự thêm trong ngoặc. KHÔNG bỏ dấu tiếng Việt ("Bánh bò" khác "Bánh bơ")
    và không dùng từ điển từ bỏ qua (trước đây "Bia Tiger" ~ "Tiger" dựa trên danh sách từ ngành hàng).
    """
    if a1 is None or a2 is None:
        return False
    c1 = _scorer_norm(strip_trailing_measurement_in_parens(unicodedata.normalize("NFC", str(a1)).strip()))
    c2 = _scorer_norm(strip_trailing_measurement_in_parens(unicodedata.normalize("NFC", str(a2)).strip()))
    return bool(c1) and c1 == c2


def check_consensus(pred1: Dict[str, Any], pred2: Dict[str, Any]) -> bool:
    """
    Kiểm tra tính đồng thuận giữa hai lời giải độc lập (Dual-Prompt Consensus Gate).
    Chỉ trả về True khi cả hai lời giải ra CÙNG MỘT ĐÁP ÁN (cùng type, cùng giá trị sau khi làm tròn chuẩn).
    Nếu bất kỳ bên nào lỗi, abstain hoặc vênh nhau -> trả về False để kích hoạt Abstain an toàn.
    """
    if not isinstance(pred1, dict) or not isinstance(pred2, dict):
        return False

    t1 = str(pred1.get("type", "")).strip().lower()
    t2 = str(pred2.get("type", "")).strip().lower()

    if not t1 or not t2:
        return False

    # Nếu một trong hai bên là abstain -> không có đồng thuận hợp lệ
    if t1 == "abstain" or t2 == "abstain":
        return False

    # Cả hai đều kết luận không thể trả lời -> Đồng thuận
    if t1 == "unanswerable" and t2 == "unanswerable":
        return True
    if t1 == "unanswerable" or t2 == "unanswerable":
        return False

    # Khác type -> Bất đồng
    if t1 != t2:
        return False

    a1 = pred1.get("answer")
    a2 = pred2.get("answer")
    if a1 is None or a2 is None:
        return False

    # 1. So khớp NUMBER hoặc PERCENT
    if t1 in ("number", "percent"):
        try:
            f1 = float(a1)
            f2 = float(a2)
            if math.isnan(f1) or math.isnan(f2) or math.isinf(f1) or math.isinf(f2):
                return False
            # Khớp tuyệt đối hoặc sai số tương đối cực nhỏ cho số tài chính lớn
            diff = abs(f1 - f2)
            max_val = max(abs(f1), abs(f2))
            if diff < 1e-4 or (max_val > 0 and diff / max_val < 1e-5):
                return True
            # Post-Rounding Consensus: chỉ so ở độ chính xác chấm điểm (2 chữ số thập phân).
            # Không so sau khi làm tròn về số nguyên: 12.4% và 12.0% (hay TB 2.45 và 2.1) là hai đáp án khác nhau.
            # Tiền đã được postprocess làm tròn về đồng trước khi vào đây nên vẫn khớp đúng.
            return round_half_up_2dec(f1) == round_half_up_2dec(f2)
        except (ValueError, TypeError):
            return False

    # 2. So khớp DATE
    if t1 == "date":
        from src.ingest import normalize_date_value_iso
        d1_norm = normalize_date_value_iso(str(a1).strip())
        d2_norm = normalize_date_value_iso(str(a2).strip())
        if d1_norm and d2_norm:
            return d1_norm == d2_norm
        return str(a1).strip() == str(a2).strip()

    # 3. So khớp BOOLEAN
    if t1 == "boolean":
        return bool(a1) == bool(a2)

    # 4. So khớp STRING
    if t1 == "string":
        return are_strings_in_consensus(a1, a2)

    # 5. So khớp LIST
    if t1 == "list":
        if not isinstance(a1, list) or not isinstance(a2, list):
            return False
        # Nếu cùng độ dài -> so khớp từng phần tử
        if len(a1) == len(a2):
            for x, y in zip(a1, a2):
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    diff = abs(float(x) - float(y))
                    max_v = max(abs(float(x)), abs(float(y)))
                    if diff >= 1e-4 and (max_v == 0 or diff / max_v >= 1e-5):
                        return False
                else:
                    if not are_strings_in_consensus(x, y):
                        return False
            return True
        # Khác độ dài -> bất đồng: list chấm đúng cả tập lẫn thứ tự, danh sách ngắn hơn là đáp án khác.
        # (Câu hỏi top k đã được DataMateAgent cắt về k phần tử trước khi so.)
        return False

    # 6. So khớp TABLE
    if t1 == "table":
        if not isinstance(a1, dict) or not isinstance(a2, dict):
            return False
        r1 = a1.get("rows")
        r2 = a2.get("rows")
        if not isinstance(r1, list) or not isinstance(r2, list):
            return False
        if len(r1) != len(r2) or len(r1) == 0:
            return False

        def _match_table_cells(c1: Any, c2: Any) -> bool:
            if isinstance(c1, (int, float)) and isinstance(c2, (int, float)):
                diff = abs(float(c1) - float(c2))
                max_v = max(abs(float(c1)), abs(float(c2)))
                return diff < 1e-4 or (max_v > 0 and diff / max_v < 1e-5)
            return are_strings_in_consensus(c1, c2)

        def _match_rows(row1: List[Any], row2: List[Any]) -> bool:
            if len(row1) != len(row2):
                return False
            return all(_match_table_cells(c1, c2) for c1, c2 in zip(row1, row2))

        # Thứ tự các dòng không ảnh hưởng kết quả chấm (Rulebook Mục 9): Bipartite matching
        used_indices = set()
        for row1 in r1:
            matched = False
            for j, row2 in enumerate(r2):
                if j in used_indices:
                    continue
                if _match_rows(row1, row2):
                    used_indices.add(j)
                    matched = True
                    break
            if not matched:
                return False
        return True

    return False
