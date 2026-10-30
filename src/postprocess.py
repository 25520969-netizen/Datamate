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
)

NON_NEGATIVE_KEYWORDS = (
    "tổng doanh thu",
    "bán được bao nhiêu",
    "bán đc tổng cộng bao nhiêu",
    "có bao nhiêu đơn",
    "co bao nhiu don",
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


def postprocess_and_guard(
    question_id: str,
    question: str,
    raw_output: Any,
    expected_type: str = "number",
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

    if isinstance(raw_output, dict) and "type" in raw_output:
        ans_type = str(raw_output.get("type", "")).strip().lower()
        ans_val = raw_output.get("answer")
        if ans_val is None:
            if "columns" in raw_output and "rows" in raw_output:
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
    if isinstance(ans_val, dict) and "columns" in ans_val and "rows" in ans_val:
        ans_type = "table"
    elif isinstance(ans_val, list) and len(ans_val) > 1 and ans_type not in ("list", "table"):
        ans_type = "list"
    elif isinstance(ans_val, bool):
        ans_type = "boolean"

    # Tự động unpack nếu kết quả trả về list 1 phần tử nhưng kiểu kỳ vọng là scalar
    if ans_type in ("number", "percent", "string", "date", "boolean") and isinstance(ans_val, list) and len(ans_val) == 1:
        ans_val = ans_val[0]

    # 1. Xử lý NUMBER
    if ans_type == "number":
        q_low = question.lower()
        if isinstance(ans_val, bool) or not isinstance(ans_val, (int, float)):
            return abstain_resp
        if math.isnan(ans_val) or math.isinf(ans_val):
            return abstain_resp

        if is_Strictly_non_negative_question(question) and float(ans_val) < 0:
            return abstain_resp

        if is_money_question(question) or float(ans_val).is_integer():
            final_num: Any = round_half_up_int(float(ans_val))
        else:
            final_num = round_half_up_2dec(float(ans_val))

        # Sanity Guard 1: Nhầm số lượng thành doanh thu
        # Câu hỏi hỏi số lượng sản phẩm/nồi/đĩa/kg/ly/chai/hộp/lon mà kết quả > 20,000 (do model sum Thành tiền)
        if any(k in q_low for k in ["mấy nồi", "mấy đĩa", "mấy ly", "mấy lon", "mấy kg", "mấy chai", "mấy hộp", "may noi", "may dia", "may ly", "may lon", "may kg", "may chai", "may hop", "bao nhiêu nồi", "bao nhiêu đĩa", "bao nhiêu ly", "bao nhiêu lon", "bao nhiêu kg", "bao nhiêu chai", "bao nhiêu hộp", "số lượng"]) and not any(k in q_low for k in ["tiền", "đồng", "doanh thu", "dthu", "giá"]):
            if final_num > 20000:
                return abstain_resp

        # Sanity Guard 2: Lọc sót ngày dẫn tới cộng dồn toàn bộ doanh thu shop (> 500 triệu cho 1 ngày duy nhất)
        if any(k in q_low for k in ["hôm nay", "hnay", "hom ni", "hôm qua", "hom qua", "bữa qua", "bua qua", "bữa nay", "bua nay"]):
            if final_num > 500000000:
                return abstain_resp

        # Sanity Guard 3: Nếu câu hỏi về doanh thu / số đơn phát sinh trong ngày/tuần/tháng đang hoạt động mà ra 0
        # thì 99% là lỗi lọc ngày rỗng -> kích hoạt abstain (0 điểm) để tránh nhận điểm phạt (-1 điểm)
        if final_num == 0:
            if any(k in q_low for k in ["hôm nay", "hnay", "hom ni", "bữa nay", "bua nay", "riêng hôm nay", "doanh thu hôm", "dthu hôm", "được mấy đơn", "mấy đh"]):
                return abstain_resp

        return {"id": question_id, "type": "number", "answer": final_num}

    # 2. Xử lý PERCENT
    if ans_type == "percent":
        q_low = question.lower()
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

        # Sanity check: tỉ lệ phần trăm không thể < -100% (trừ tăng trưởng giảm tối đa 100%)
        if float(ans_val) < -100.0:
            return abstain_resp

        final_pct = round_half_up_2dec(float(ans_val))

        # Sanity Guard 4: Tỉ trọng / Tỉ lệ đóng góp không thể vượt quá 100% hoặc nhỏ hơn 0%
        if any(k in q_low for k in ["chiếm", "chiem", "tỉ trọng", "tỷ trọng", "ti trong", "đóng góp", "dong gop", "tỉ lệ đơn hủy", "tỷ lệ đơn hủy", "tỉ lệ hủy", "tỷ lệ huỷ"]):
            if final_pct > 100.0 or final_pct < 0.0:
                return abstain_resp

        # Sanity Guard 5: Tăng/giảm hoặc thay đổi % ra đúng 0.0% thường do lọc ngày bị rỗng
        if final_pct == 0.0:
            if any(k in q_low for k in ["thay đổi", "thay doi", "tăng hay giảm", "tang hay giam", "tăng giảm", "tang giam"]):
                return abstain_resp

        return {"id": question_id, "type": "percent", "answer": final_pct}

    # 3. Xử lý DATE
    if ans_type == "date":
        norm_dt = normalize_date_str(ans_val)
        if not norm_dt:
            return abstain_resp
        return {"id": question_id, "type": "date", "answer": norm_dt}

    # 4. Xử lý BOOLEAN
    if ans_type == "boolean":
        if not isinstance(ans_val, bool):
            return abstain_resp
        return {"id": question_id, "type": "boolean", "answer": ans_val}

    # 5. Xử lý STRING
    if ans_type == "string":
        if not isinstance(ans_val, str):
            return abstain_resp
        clean_str = unicodedata.normalize("NFC", ans_val).strip()

        # Nếu model trả về chuỗi ghép nhiều đối tượng bằng dấu phẩy mà câu hỏi hỏi danh sách/top N
        # -> Tự động chuyển đổi thành type "list" chuẩn JSON
        if "," in clean_str and (expected_type == "list" or any(kw in question.lower() for kw in ["top", "danh sách", "liệt kê", "các", "những"])):
            parts = [
                re.sub(r"\s*\([^)]*\)", "", unicodedata.normalize("NFC", p)).strip()
                for p in clean_str.split(",")
                if p.strip()
            ]
            parts = [p for p in parts if p.lower() not in ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai")]
            if len(parts) > 1:
                return {"id": question_id, "type": "list", "answer": parts}

        # Loại bỏ các số đo / số tiền trong ngoặc đơn (vd: 'KH0014 (1,605,000)' -> 'KH0014', 'Trà đào (307 ly)' -> 'Trà đào')
        clean_str = re.sub(r"\s*\([^)]*\)", "", clean_str).strip()

        # Mục 3 Rulebook: Khách lẻ / kl không phải khách, không xếp hạng
        if clean_str.lower() in ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai"):
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
                    re.sub(r"\s*\([^)]*\)", "", re.sub(r"^\s*[-*•\d\.\)]+\s*", "", p)).strip()
                    for p in parts
                ]
                ans_val = [p for p in ans_val if p and p.lower() not in ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai")]

        if not isinstance(ans_val, list) or len(ans_val) == 0:
            return abstain_resp
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
                # Loại bỏ số tiền/số lượng trong ngoặc đơn của item
                s_item = re.sub(r"\s*\([^)]*\)", "", s_item).strip()
                # Khử khách lẻ
                if s_item.lower() in ("khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai", "none", "nan", "null"):
                    continue
                # Tránh chuỗi rác dài bất thường
                if len(s_item) > 120:
                    return abstain_resp
                if s_item:
                    cleaned_list.append(s_item)
            else:
                return abstain_resp

        # Khử trùng lặp liên tiếp giữ nguyên thứ tự
        seen = set()
        deduped = []
        for x in cleaned_list:
            if x not in seen:
                seen.add(x)
                deduped.append(x)

        if not deduped:
            return abstain_resp
        return {"id": question_id, "type": "list", "answer": deduped}

    # 7. Xử lý TABLE
    if ans_type == "table":
        if not isinstance(ans_val, dict):
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
            for idx, cell in enumerate(row):
                if cell is None:
                    return abstain_resp
                if idx == 0 and isinstance(cell, str):
                    s_cell = unicodedata.normalize("NFC", cell).strip()
                    if not s_cell:
                        return abstain_resp
                    c_row.append(s_cell)
                elif isinstance(cell, (int, float)) and not isinstance(cell, bool):
                    if math.isnan(cell) or math.isinf(cell):
                        return abstain_resp
                    if float(cell).is_integer():
                        c_row.append(round_half_up_int(float(cell)))
                    else:
                        c_row.append(round_half_up_2dec(float(cell)))
                elif isinstance(cell, str):
                    c_row.append(unicodedata.normalize("NFC", cell).strip())
                else:
                    return abstain_resp
            clean_rows.append(c_row)

        return {
            "id": question_id,
            "type": "table",
            "answer": {"columns": clean_cols, "rows": clean_rows},
        }

    return abstain_resp
