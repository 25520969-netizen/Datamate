# -*- coding: utf-8 -*-
"""
Tạo file chi tiết dễ đọc (detail.json) từ log từng câu của pipeline: mô tả rõ quyết định của Prompt 1, Prompt 2
(và Prompt 3 nếu có bỏ phiếu), lý do chốt đáp án cuối, và (với tập train) đối chiếu đáp án chuẩn.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evaluate import score_single_prediction

REASON_TEXT = {
    "consensus_match": "Prompt 1 và Prompt 2 cho CÙNG một đáp án nên được chấp nhận (đồng thuận).",
    "deterministic_unanswerable": "Luật tất định (kiểm tra schema/khoảng ngày) kết luận dữ liệu không đủ để trả lời, "
                                  "nên trả unanswerable mà không cần gọi mô hình.",
    "consensus_disagreement_abstain": "Hai phương án (và phiếu thứ ba nếu có) KHÔNG đồng thuận, không có căn cứ chọn bên nào "
                                      "nên abstain để tránh bị trừ điểm.",
    "table_disagreement_abstain": "Hai bảng kết quả khác nhau, không phân xử được nên abstain.",
    "strict_requires_both": "Chỉ một prompt cho đáp án hợp lệ; chế độ strict yêu cầu cả hai nên abstain.",
    "p2_only_rejected_abstain": "Chỉ Prompt 2 cho đáp án hợp lệ; chế độ strict yêu cầu cả hai nên abstain.",
    "both_failed": "Cả hai prompt đều không cho đáp án hợp lệ (lỗi chạy mã, kết quả rỗng hoặc bị bộ kiểm tra loại) nên abstain.",
    "exception_before_prompt": "Lỗi ở bước chuẩn bị trước khi gọi mô hình nên abstain.",
    "unhandled_exception": "Lỗi bất ngờ khi xử lý câu hỏi nên abstain.",
    "single_prompt_mode": "Chế độ một prompt: dùng đáp án của Prompt 1.",
    "p1_valid_p2_failed_fallback": "Prompt 2 lỗi, Prompt 1 hợp lệ: giữ đáp án Prompt 1 (chế độ adaptive/tournament).",
    "p2_valid_p1_failed_fallback": "Prompt 1 lỗi, Prompt 2 hợp lệ: giữ đáp án Prompt 2 (chế độ adaptive/tournament).",
    "p1_distinct_semantic_arbitration": "Câu hỏi đếm đơn: Prompt 1 dùng COUNT(DISTINCT ...) còn Prompt 2 thì không, chọn Prompt 1.",
    "p2_distinct_semantic_arbitration": "Câu hỏi đếm đơn: Prompt 2 dùng COUNT(DISTINCT ...) còn Prompt 1 thì không, chọn Prompt 2.",
    "retail_customer_arbitration_p1": "Prompt 2 trả về nhãn khách lẻ (không phải khách theo rulebook), chọn đáp án Prompt 1.",
    "retail_customer_arbitration_p2": "Prompt 1 trả về nhãn khách lẻ (không phải khách theo rulebook), chọn đáp án Prompt 2.",
    "tournament_referee_p1": "Trọng tài LLM thấy Prompt 2 vi phạm rulebook, chọn Prompt 1.",
    "tournament_referee_p2": "Trọng tài LLM thấy Prompt 1 vi phạm rulebook, chọn Prompt 2.",
}


def _short(v: Any, n: int = 160) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def explain_prompt(name: str, code: Any, raw: Any, pred: Any) -> Dict[str, Any]:
    """Mô tả kết quả của một prompt: có chạy được không, đáp án thô, đáp án sau chuẩn hoá."""
    if code in (None, "") and raw is None and pred is None:
        return {"ran": False, "status": "không chạy", "note": f"{name} không được chạy cho câu này."}
    ptype = pred.get("type") if isinstance(pred, dict) else None
    answer = pred.get("answer") if isinstance(pred, dict) else None
    error = pred.get("error") if isinstance(pred, dict) else None
    if isinstance(code, str) and code.startswith("# Deterministic"):
        status, note = "bỏ qua", "Câu đã được kết luận unanswerable bằng luật tất định trước khi gọi mô hình."
    elif ptype == "abstain" or ptype is None:
        status = "không dùng được"
        note = error or "Mã chạy lỗi, kết quả rỗng/không hợp lệ hoặc bị bộ kiểm tra loại."
    elif ptype == "unanswerable":
        status, note = "unanswerable", "Mô hình kết luận dữ liệu không đủ để trả lời."
    else:
        status, note = "hợp lệ", f"Đáp án kiểu {ptype}: {_short(answer)}"
    return {
        "ran": True,
        "status": status,
        "note": note,
        "answer_type": ptype,
        "answer": answer,
        "raw_sandbox_result": raw,
        "generated_code": code,
    }


def describe_decision(d: Dict[str, Any]) -> str:
    reason = d.get("decision_reason", "")
    base = REASON_TEXT.get(reason)
    if base is None and str(reason).startswith("majority_vote_"):
        who = str(reason)[len("majority_vote_"):].replace("_", " + ").upper()
        base = (f"Prompt 1 và Prompt 2 bất đồng nên chạy Prompt 3 làm lá phiếu thứ ba; đáp án của nhóm {who} "
                "(>= 2/3 phiếu) được chọn.")
    base = base or f"Quyết định theo luật '{reason}'."
    final = d.get("final_decision")
    if final == "abstain":
        return base + " → KẾT QUẢ: abstain (0 điểm)."
    if final == "unanswerable":
        return base + " → KẾT QUẢ: unanswerable."
    return base + f" → KẾT QUẢ: {final} = {_short(d.get('final_answer'))}."


def build_detail_records(
    details: Iterable[Optional[Dict[str, Any]]],
    questions: List[Dict[str, Any]],
    predictions: Optional[List[Optional[Dict[str, Any]]]] = None,
    gold_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Một bản ghi đọc được cho mỗi câu, theo đúng thứ tự file câu hỏi."""
    out: List[Dict[str, Any]] = []
    details = list(details)
    for i, q in enumerate(questions):
        d = details[i] if i < len(details) else None
        if d is None:
            out.append({"id": q["id"], "business_id": q.get("business_id"), "question": q.get("question"),
                        "decision_summary": "Chưa chạy xong câu này (tính là abstain)."})
            continue
        rec: Dict[str, Any] = {
            "id": q["id"],
            "business_id": q.get("business_id"),
            "as_of_date": q.get("as_of_date"),
            "question": q.get("question"),
            "question_after_normalization": d.get("question"),
            "expected_type": d.get("expected_type"),
            "decision_summary": describe_decision(d),
            "decision_reason": d.get("decision_reason"),
            "is_consensus": d.get("is_consensus"),
            "final": {"type": d.get("final_decision"), "answer": d.get("final_answer")},
            "prompt1": explain_prompt("Prompt 1", d.get("prompt1_code"), d.get("prompt1_raw"), d.get("prompt1_pred")),
            "prompt2": explain_prompt("Prompt 2", d.get("prompt2_code"), d.get("prompt2_raw"), d.get("prompt2_pred")),
        }
        if d.get("prompt3_pred") is not None or d.get("prompt3_code"):
            rec["prompt3_tiebreak"] = explain_prompt("Prompt 3", d.get("prompt3_code"), d.get("prompt3_raw"), d.get("prompt3_pred"))
        if d.get("error"):
            rec["error"] = d["error"]
        if gold_map is not None:
            g = gold_map.get(q["id"])
            pred = predictions[i] if predictions is not None and i < len(predictions) else None
            if g is not None:
                score, status = score_single_prediction(pred, g)
                rec["gold"] = {"type": g.get("type"), "answer": g.get("answer")}
                rec["verdict"] = {"correct": "ĐÚNG (+1)", "wrong": "SAI (-1)", "abstain": "BỎ QUA (0)"}[status]
                rec["score"] = score
        out.append(rec)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)
