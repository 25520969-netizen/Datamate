#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bộ chấm điểm chuẩn (Evaluator) cho bài toán DataMate - Zalo AI Challenge 2026.
Tuân thủ tuyệt đối Mục 7 và Mục 9 của train/rulebook.md:
- Đúng: +1 điểm
- Sai: -1 điểm
- Bỏ qua (abstain hoặc vắng mặt): 0 điểm
- Penalized Accuracy (PA) = (Correct - Wrong) / Total
"""

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_TYPES = {
    "number",
    "percent",
    "date",
    "boolean",
    "string",
    "list",
    "table",
    "unanswerable",
    "abstain",
}


def is_json_number(val: Any) -> bool:
    """Kiểm tra giá trị là JSON number hợp lệ (int/float, không phải bool, không phải NaN/Inf)."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def normalize_string(text: str) -> str:
    """
    Chuẩn hóa chuỗi theo Mục 9 Rulebook:
    - Chuẩn hóa Unicode NFC (giữ nguyên dấu tiếng Việt).
    - Không phân biệt hoa thường.
    - Không phân biệt khoảng trắng và dấu câu.
    """
    if not isinstance(text, str):
        return ""
    nfc = unicodedata.normalize("NFC", text).lower().strip()
    # Loại bỏ dấu câu (giữ lại chữ cái có dấu tiếng Việt và chữ số)
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in nfc)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_string_compact(text: str) -> str:
    """Chuẩn hóa chuỗi không khoảng trắng để khớp các trường hợp như '1000mg' và '1000 mg'."""
    return normalize_string(text).replace(" ", "")


def compare_numbers(pred: Any, gold: Any, tol: float = 1e-5) -> bool:
    """So sánh hai số JSON, chỉ bỏ qua sai số dấu phẩy động."""
    if not is_json_number(pred) or not is_json_number(gold):
        return False
    return abs(float(pred) - float(gold)) <= tol


def compare_strings(pred: Any, gold: Any, aliases: Optional[List[str]] = None) -> bool:
    """So sánh chuỗi dự đoán với đáp án chuẩn và danh sách alias."""
    if not isinstance(pred, str) or not isinstance(gold, str):
        return False
    candidates = [gold]
    if aliases:
        candidates.extend(a for a in aliases if isinstance(a, str))

    pred_norm = normalize_string(pred)
    pred_compact = normalize_string_compact(pred)
    if not pred_norm:
        return normalize_string(gold) == ""

    for cand in candidates:
        if pred_norm == normalize_string(cand) or pred_compact == normalize_string_compact(cand):
            return True
    return False


def compare_dates(pred: Any, gold: Any) -> bool:
    """
    Kiểm tra định dạng date chuẩn ISO ('YYYY-MM-DD' hoặc 'YYYY-MM') và so khớp.
    """
    if not isinstance(pred, str) or not isinstance(gold, str):
        return False
    pred_clean = pred.strip()
    gold_clean = gold.strip()
    if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", pred_clean):
        return False
    return pred_clean == gold_clean


def compare_lists(pred: Any, gold: Any, aliases: Optional[List[Any]] = None) -> bool:
    """
    So sánh mảng JSON có thứ tự theo Mục 9 Rulebook.
    Mỗi phần tử thứ i có thể có danh sách alias tại aliases[i].
    """
    if not isinstance(pred, list) or not isinstance(gold, list):
        return False
    if len(pred) != len(gold):
        return False

    for idx, (p_item, g_item) in enumerate(zip(pred, gold)):
        item_aliases: Optional[List[str]] = None
        if isinstance(aliases, list) and idx < len(aliases):
            if isinstance(aliases[idx], list):
                item_aliases = aliases[idx]
            elif isinstance(aliases[idx], str):
                item_aliases = [aliases[idx]]

        if is_json_number(g_item):
            if not compare_numbers(p_item, g_item):
                return False
        elif isinstance(g_item, str):
            if not compare_strings(p_item, g_item, item_aliases):
                return False
        elif isinstance(g_item, bool):
            if not (isinstance(p_item, bool) and p_item == g_item):
                return False
        else:
            if p_item != g_item:
                return False

    return True


def _match_table_row(
    pred_row: List[Any],
    gold_row: List[Any],
    row_aliases: Optional[List[str]],
    col_mapping: List[int],
) -> bool:
    """Kiểm tra một dòng của bảng dự đoán có khớp với dòng đáp án chuẩn hay không."""
    if not isinstance(pred_row, list) or len(pred_row) != len(gold_row):
        return False

    for g_col_idx, g_val in enumerate(gold_row):
        p_col_idx = col_mapping[g_col_idx]
        p_val = pred_row[p_col_idx]

        if g_col_idx == 0 and isinstance(g_val, str):
            if not compare_strings(p_val, g_val, row_aliases):
                return False
        elif is_json_number(g_val):
            if not compare_numbers(p_val, g_val):
                return False
        elif isinstance(g_val, str):
            if not compare_strings(p_val, g_val):
                return False
        elif isinstance(g_val, bool):
            if not (isinstance(p_val, bool) and p_val == g_val):
                return False
        else:
            if p_val != g_val:
                return False

    return True


def compare_tables(pred: Any, gold: Any, aliases: Optional[List[Any]] = None) -> bool:
    """
    So sánh bảng theo Mục 9 Rulebook:
    - Cấu trúc: {"columns": [...], "rows": [[...], ...]}
    - Cột so theo tên; tên không khớp thì so theo vị trí.
    - Thứ tự các dòng không ảnh hưởng kết quả chấm.
    """
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return False
    pred_cols = pred.get("columns")
    pred_rows = pred.get("rows")
    gold_cols = gold.get("columns")
    gold_rows = gold.get("rows")

    if not isinstance(pred_cols, list) or not isinstance(pred_rows, list):
        return False
    if not isinstance(gold_cols, list) or not isinstance(gold_rows, list):
        return False
    if len(pred_cols) != len(gold_cols) or len(pred_rows) != len(gold_rows):
        return False

    # Ánh xạ cột theo tên nếu khớp hoàn toàn, ngược lại so theo vị trí
    pred_col_norms = [normalize_string(str(c)) for c in pred_cols]
    gold_col_norms = [normalize_string(str(c)) for c in gold_cols]
    if set(pred_col_norms) == set(gold_col_norms) and len(set(gold_col_norms)) == len(gold_cols):
        col_mapping = [pred_col_norms.index(gc) for gc in gold_col_norms]
    else:
        col_mapping = list(range(len(gold_cols)))

    # Kiểm tra mọi ô số trong pred_rows tương ứng với cột số của gold phải là JSON number
    used_pred_indices = set()
    for r_idx, g_row in enumerate(gold_rows):
        row_aliases: Optional[List[str]] = None
        if isinstance(aliases, list) and r_idx < len(aliases):
            if isinstance(aliases[r_idx], list):
                row_aliases = aliases[r_idx]
            elif isinstance(aliases[r_idx], str):
                row_aliases = [aliases[r_idx]]

        matched = False
        for p_idx, p_row in enumerate(pred_rows):
            if p_idx in used_pred_indices:
                continue
            if _match_table_row(p_row, g_row, row_aliases, col_mapping):
                used_pred_indices.add(p_idx)
                matched = True
                break
        if not matched:
            return False

    return True


def score_single_prediction(pred_obj: Optional[Dict[str, Any]], gold_obj: Dict[str, Any]) -> Tuple[int, str]:
    """
    Chấm điểm một câu hỏi đơn lẻ.
    Trả về: (điểm_số, trạng_thái)
      - (0, "abstain") nếu bỏ trống hoặc type == "abstain"
      - (1, "correct") nếu đúng cả type, format và giá trị
      - (-1, "wrong") nếu sai
    """
    if pred_obj is None or not isinstance(pred_obj, dict):
        return 0, "abstain"

    pred_type = pred_obj.get("type")
    if pred_type is None or pred_type == "abstain":
        return 0, "abstain"

    if pred_type not in VALID_TYPES:
        return -1, "wrong"

    gold_type = gold_obj.get("type")
    gold_ans = gold_obj.get("answer")
    gold_aliases = gold_obj.get("alias", [])

    # Mục 7 Rulebook: Với unanswerable, nếu có trường reason mà không phải chuỗi -> tính như bỏ trống (0 điểm)
    if pred_type == "unanswerable" and "reason" in pred_obj:
        reason_val = pred_obj["reason"]
        if reason_val is not None and not isinstance(reason_val, str):
            return 0, "abstain"

    # Khác type là sai (-1 điểm)
    if pred_type != gold_type:
        return -1, "wrong"

    if gold_type == "unanswerable":
        return 1, "correct"

    pred_ans = pred_obj.get("answer")

    if gold_type in ("number", "percent"):
        is_ok = compare_numbers(pred_ans, gold_ans)
    elif gold_type == "boolean":
        is_ok = isinstance(pred_ans, bool) and pred_ans == gold_ans
    elif gold_type == "date":
        is_ok = compare_dates(pred_ans, gold_ans)
    elif gold_type == "string":
        str_aliases = [a for a in gold_aliases if isinstance(a, str)] if isinstance(gold_aliases, list) else []
        is_ok = compare_strings(pred_ans, gold_ans, str_aliases)
    elif gold_type == "list":
        is_ok = compare_lists(pred_ans, gold_ans, gold_aliases)
    elif gold_type == "table":
        is_ok = compare_tables(pred_ans, gold_ans, gold_aliases)
    else:
        is_ok = False

    return (1, "correct") if is_ok else (-1, "wrong")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Đọc file JSONL, bỏ qua các dòng không hợp lệ."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "id" in obj:
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def evaluate_predictions(
    pred_records: List[Dict[str, Any]],
    gold_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tính toán toàn bộ chỉ số Penalized Accuracy trên danh sách dự đoán và đáp án chuẩn.
    """
    pred_map: Dict[str, Dict[str, Any]] = {r["id"]: r for r in pred_records if "id" in r}
    total = len(gold_records)
    correct = 0
    wrong = 0
    abstain = 0
    by_type: Dict[str, Dict[str, int]] = {}
    errors: List[Dict[str, Any]] = []

    for gold in gold_records:
        qid = gold["id"]
        g_type = gold.get("type", "unknown")
        if g_type not in by_type:
            by_type[g_type] = {"total": 0, "correct": 0, "wrong": 0, "abstain": 0}
        by_type[g_type]["total"] += 1

        pred = pred_map.get(qid)
        score, status = score_single_prediction(pred, gold)
        if status == "correct":
            correct += 1
            by_type[g_type]["correct"] += 1
        elif status == "wrong":
            wrong += 1
            by_type[g_type]["wrong"] += 1
            errors.append(
                {
                    "id": qid,
                    "gold_type": g_type,
                    "gold_answer": gold.get("answer"),
                    "pred_type": pred.get("type") if pred else None,
                    "pred_answer": pred.get("answer") if pred else None,
                }
            )
        else:
            abstain += 1
            by_type[g_type]["abstain"] += 1

    raw_points = correct - wrong
    pa_score = (raw_points / total) if total > 0 else 0.0
    accuracy = (correct / total) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "raw_points": raw_points,
        "penalized_accuracy": round(pa_score, 6),
        "accuracy": round(accuracy, 6),
        "by_type": by_type,
        "sample_errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chấm điểm DataMate theo chuẩn Penalized Accuracy.")
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="Đường dẫn tới file submission.jsonl cần chấm.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(__file__).resolve().parent / "train" / "answers.jsonl",
        help="Đường dẫn tới file answers.jsonl chuẩn.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Đường dẫn lưu báo cáo JSON chi tiết (tùy chọn).",
    )
    args = parser.parse_args()

    gold_records = load_jsonl(args.gold)
    pred_records = load_jsonl(args.pred)
    report = evaluate_predictions(pred_records, gold_records)

    print("=" * 60)
    print("KẾT QUẢ CHẤM ĐIỂM DATAMATE (PENALIZED ACCURACY)")
    print("=" * 60)
    print(f"Tổng số câu hỏi (N)      : {report['total']}")
    print(f"Trả lời ĐÚNG (+1)        : {report['correct']}")
    print(f"Trả lời SAI  (-1)        : {report['wrong']}")
    print(f"Bỏ qua / Abstain (0)     : {report['abstain']}")
    print(f"Tổng điểm (Correct-Wrong): {report['raw_points']}")
    print(f"Penalized Accuracy (PA)  : {report['penalized_accuracy'] * 100:.2f}%")
    print("-" * 60)
    print("Chi tiết theo từng kiểu đáp án (type):")
    for t_name, stats in sorted(report["by_type"].items()):
        t_pa = (stats["correct"] - stats["wrong"]) / stats["total"] * 100 if stats["total"] else 0.0
        print(
            f"  - {t_name:<14}: Total={stats['total']:<4} | "
            f"Correct={stats['correct']:<4} | Wrong={stats['wrong']:<4} | "
            f"Abstain={stats['abstain']:<4} | PA={t_pa:6.2f}%"
        )
    print("=" * 60)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu báo cáo chi tiết tại: {args.output}")


if __name__ == "__main__":
    main()
