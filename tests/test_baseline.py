# -*- coding: utf-8 -*-
"""
Unit tests cho model/baseline.py: DataMateAgent & Trọng tài Safe Abstain.
Kiểm tra nguyên tắc: khi 2 Prompt bất đồng kết quả mà không có căn cứ ngữ nghĩa rõ ràng,
hệ thống BẮT BUỘC trả về abstain để bảo vệ điểm số trên Penalized Accuracy (PA).
"""

from pathlib import Path
import pytest
from model.baseline import DataMateAgent, QwenCoderClient


class MockLLMClient:
    def __init__(self, p1_reply, p2_reply):
        self.p1_reply = p1_reply
        self.p2_reply = p2_reply
        self.call_count = 0
        self.backend = "mock"
        self.model_name = "mock_model"

    def health_check(self, timeout=5.0):
        return True

    def generate(self, messages, seed=42):
        self.call_count += 1
        # Lần 1 là P1 (seed 42), lần 2 là P2 (seed 100)
        if seed == 42:
            return self.p1_reply
        return self.p2_reply


def test_baseline_consensus_accepts_matching_answers(monkeypatch, tmp_path):
    # Cả P1 và P2 đều ra 5000000 -> Đồng thuận
    code1 = "```python\nresult = {'type': 'number', 'answer': 5000000}\n```"
    code2 = "```python\nresult = {'type': 'number', 'answer': 5000000}\n```"
    client = MockLLMClient(p1_reply=code1, p2_reply=code2)

    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )

    # Mock các tầng trước để test riêng trọng tài
    q_obj = {"id": "test_q1", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Doanh thu bao nhiêu?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))
    monkeypatch.setattr(agent.sandbox, "execute", lambda code, **kw: {"ok": True, "result": {"type": "number", "answer": 5000000}})

    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "number"
    assert pred["answer"] == 5000000
    assert details["is_consensus"] is True


def test_baseline_safe_abstain_on_disagreement(monkeypatch, tmp_path):
    # P1 ra 5,000,000, P2 ra 8,000,000 -> Bất đồng -> PHẢI ABSTAIN
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")

    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )

    q_obj = {"id": "test_q2", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Doanh thu bao nhiêu?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    # Giả lập P1 trả 5M, P2 trả 8M
    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "number", "answer": 5000000}, "code1", 5000000
        return {"id": qid, "type": "number", "answer": 8000000}, "code2", 8000000

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)

    pred, details = agent.answer_question_dual(q_obj)
    # Tuyệt đối không được ép nhận P1! Phải trả về abstain để tránh bị phạt -1 điểm!
    assert pred["type"] == "abstain"
    assert details["is_consensus"] is False
    assert details["decision_reason"] == "consensus_disagreement_abstain"


def test_baseline_retail_customer_arbitration(monkeypatch, tmp_path):
    # Rulebook Mục 3: Khách lẻ không phải khách, không đếm và không xếp hạng.
    # Nếu P2 ra "Khách lẻ", P1 ra "Nguyễn Văn A" -> Ưu tiên P1
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")

    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )

    q_obj = {"id": "test_q3", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Khách hàng nào mua nhiều tiền nhất?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "string", "answer": "Nguyễn Văn A"}, "code1", {"type": "string", "answer": "Nguyễn Văn A"}
        return {"id": qid, "type": "string", "answer": "Khách lẻ"}, "code2", {"type": "string", "answer": "Khách lẻ"}

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)

    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "string"
    assert pred["answer"] == "Nguyễn Văn A"
    assert details["decision_reason"] == "retail_customer_arbitration_p1"


def test_baseline_safe_abstain_on_table_mismatch(monkeypatch, tmp_path):
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )
    q_obj = {"id": "test_q4", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Top sản phẩm bán chạy?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    t1 = {"columns": ["SP", "SL"], "rows": [["Trà", 10], ["Cà phê", 5]]}
    t2 = {"columns": ["SP", "SL"], "rows": [["Trà", 10], ["Cà phê", 5], ["Bánh", 2]]}

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "table", "answer": t1}, "code1", t1
        return {"id": qid, "type": "table", "answer": t2}, "code2", t2

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"
    assert details["is_consensus"] is False


def test_baseline_safe_abstain_on_type_mismatch(monkeypatch, tmp_path):
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )
    q_obj = {"id": "test_q5", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Sản phẩm nào?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "string", "answer": "Trà sữa"}, "code1", "Trà sữa"
        return {"id": qid, "type": "list", "answer": ["Trà sữa", "Cà phê"]}, "code2", ["Trà sữa", "Cà phê"]

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"


def test_baseline_both_unanswerable(monkeypatch, tmp_path):
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )
    q_obj = {"id": "test_q6", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Dữ liệu năm 2030?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        return {"id": qid, "type": "unanswerable"}, "code", "unanswerable"

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "unanswerable"
    assert details["is_consensus"] is True


def test_baseline_deterministic_unanswerable_skips_llm(monkeypatch, tmp_path):
    client = MockLLMClient(p1_reply="should_not_run", p2_reply="should_not_run")
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )
    q_obj = {"id": "test_q7", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Hỏi năm 2030?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (True, "Kỳ hỏi nằm ngoài khoảng ngày của shop."))

    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "unanswerable"
    assert client.call_count == 0
    assert details["decision_reason"] == "deterministic_unanswerable"
    assert "Kỳ hỏi nằm ngoài khoảng ngày" in pred["reason"]
