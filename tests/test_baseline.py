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
        tiebreak=False,
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
        tiebreak=False,
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
        tiebreak=False,
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


def test_qwen_coder_client_seed_passed_at_zero_temperature(monkeypatch):
    """Xác nhận seed=42 được truyền đầy đủ vào payload OpenAI và Ollama ngay cả khi temperature=0.0."""
    client_openai = QwenCoderClient(backend="openai", temperature=0.0)
    captured_openai_payload = {}

    class MockResponse:
        status_code = 200
        ok = True
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}
        def raise_for_status(self):
            pass

    def mock_post(url, json=None, headers=None, timeout=None):
        nonlocal captured_openai_payload
        captured_openai_payload = json
        return MockResponse()

    monkeypatch.setattr(client_openai._session, "post", mock_post)
    client_openai.generate([{"role": "user", "content": "hi"}], seed=42)

    assert "seed" in captured_openai_payload
    assert captured_openai_payload["seed"] == 42
    assert captured_openai_payload["temperature"] == 0.0

    # Ollama test
    client_ollama = QwenCoderClient(backend="ollama", temperature=0.0)
    captured_ollama_payload = {}

    def mock_ollama_post(url, json=None, timeout=None):
        nonlocal captured_ollama_payload
        captured_ollama_payload = json
        class MockOllamaResp:
            status_code = 200
            def json(self):
                return {"message": {"content": "ollama ok"}}
            def raise_for_status(self):
                pass
        return MockOllamaResp()

    monkeypatch.setattr(client_ollama._session, "post", mock_ollama_post)
    client_ollama.generate([{"role": "user", "content": "hi"}], seed=99)

    assert "options" in captured_ollama_payload
    assert captured_ollama_payload["options"].get("seed") == 99


def test_datamate_agent_custom_seed_propagation(monkeypatch, tmp_path):
    """Xác nhận seed cấu hình trong DataMateAgent lan truyền đúng tới P1 và P2."""
    captured_seeds = []
    class SeedCapturingClient:
        backend = "mock"
        model_name = "mock"
        def health_check(self, timeout=5.0):
            return True
        def generate(self, messages, seed=42):
            captured_seeds.append(seed)
            return "```python\nresult = {'type': 'number', 'answer': 100}\n```"

    client = SeedCapturingClient()
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        seed=77,
    )

    q_obj = {"id": "q_seed", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Hỏi seed?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))
    monkeypatch.setattr(agent.sandbox, "execute", lambda code, **kw: {"ok": True, "result": {"type": "number", "answer": 100}})

    pred, details = agent.answer_question_dual(q_obj)
    assert set(captured_seeds) == {77, 77 + 58}  # 77 cho P1, 135 cho P2


def test_baseline_safe_tournament_referee_identifies_rule_violation(monkeypatch, tmp_path):
    # P1 ra 1,000,000; P2 ra 1,500,000
    # Trong tournament mode, Reasoning Referee phát hiện P2 vi phạm (quên lọc as_of_date) -> chọn P1
    class TournamentMockClient:
        backend = "mock"
        model_name = "mock"
        def health_check(self, timeout=5.0):
            return True
        def generate(self, messages, seed=42):
            content = messages[-1]["content"]
            if "Bạn là trọng tài thẩm định" in content or "Phương án 1" in content:
                # LLM Referee phát hiện P2 vi phạm và chỉ định \boxed{1}
                return "Sau khi phân tích, Phương án 2 tính cả tương lai, Phương án 1 chuẩn xác: \\boxed{1}"
            if seed == 42:
                return "```python\nresult = {'type': 'number', 'answer': 1000000}\n```"
            return "```python\nresult = {'type': 'number', 'answer': 1500000}\n```"

    client = TournamentMockClient()
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="tournament",
        tiebreak=False,
    )
    q_obj = {"id": "q_tour1", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Doanh thu là bao nhiêu?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, max_sample_rows=0, question="": "")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "number", "answer": 1000000}, "code1", 1000000
        return {"id": qid, "type": "number", "answer": 1500000}, "code2", 1500000

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "number"
    assert pred["answer"] == 1000000
    assert details["decision_reason"] == "tournament_referee_p1"


def test_baseline_safe_tournament_referee_abstains_on_uncertainty(monkeypatch, tmp_path):
    # P1 ra 1,000,000; P2 ra 1,500,000
    # Khi Reasoning Referee không chắc chắn hoặc trả về \boxed{ABSTAIN} -> BẮT BUỘC Abstain an toàn (0 điểm)
    class UncertainTournamentClient:
        backend = "mock"
        model_name = "mock"
        def health_check(self, timeout=5.0):
            return True
        def generate(self, messages, seed=42):
            return "Cả 2 phương án đều có lý lẽ riêng, không chắc chắn: \\boxed{ABSTAIN}"

    client = UncertainTournamentClient()
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="tournament",
        tiebreak=False,
    )
    q_obj = {"id": "q_tour2", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": "Doanh thu là bao nhiêu?"}
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, max_sample_rows=0, question="": "")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        if seed == 42:
            return {"id": qid, "type": "number", "answer": 1000000}, "code1", 1000000
        return {"id": qid, "type": "number", "answer": 1500000}, "code2", 1500000

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"
    assert details["decision_reason"] == "consensus_disagreement_abstain"


def test_baseline_refinement_recovers_empty_initial_result(monkeypatch, tmp_path):
    # Lần 1: Code chạy không lỗi nhưng trả về None -> hệ thống không bỏ cuộc mà gửi feedback chẩn đoán
    # Lần 2: LLM sửa lại WHERE và trả về 500 -> thành công cứu vãn điểm số!
    call_log = []
    class RefinementClient:
        backend = "mock"
        model_name = "mock"
        def health_check(self, timeout=5.0):
            return True
        def generate(self, messages, seed=42):
            call_log.append(len(messages))
            if len(messages) == 2:
                # Lần 1: sinh code lọc quá chặt ra None
                return "```python\nresult = None\n```"
            else:
                # Lần 2 (đã có error feedback từ Sandbox): sửa lại trả về 500
                return "```python\nresult = 500\n```"

    client = RefinementClient()
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        max_retries=2,
    )

    pred, code, raw = agent._solve_single_prompt(
        system_prompt="system",
        user_prompt="user",
        qid="q_refine",
        business_id="biz_01",
        as_of_date="2026-05-15",
        date_ctx={},
        expected_type="number",
        question="Số lượng bán được?",
        seed=42,
    )
    assert pred["type"] == "number"
    assert pred["answer"] == 500
    assert len(call_log) == 2  # Đã kích hoạt đúng 2 lượt (Lần 1 rỗng -> Refine -> Lần 2 thành công)




def _make_vote_agent(monkeypatch, tmp_path, answers, question="Doanh thu bao nhiêu?"):
    """answers: dict seed -> pred cho P1 (seed 42), P2 (seed 100), P3 (seed 59)."""
    client = MockLLMClient(p1_reply="code1", p2_reply="code2")
    agent = DataMateAgent(
        data_root=tmp_path,
        calendar_path=tmp_path / "cal.csv",
        rulebook_path=tmp_path / "rb.md",
        llm_client=client,
        arbitration_strategy="strict",
    )
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "CREATE TABLE ban_hang;")
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))
    calls = []

    def mock_solve(system_prompt, user_prompt, qid, business_id, as_of_date, date_ctx, expected_type, question, seed=42):
        calls.append(seed)
        pred = answers[seed]
        return dict(pred, id=qid), f"code{seed}", pred

    monkeypatch.setattr(agent, "_solve_single_prompt", mock_solve)
    q_obj = {"id": "q_vote", "business_id": "biz_01", "as_of_date": "2026-05-15", "question": question}
    return agent, q_obj, calls


def test_tiebreak_majority_accepts_two_of_three(monkeypatch, tmp_path):
    answers = {
        42: {"type": "number", "answer": 5000000},
        100: {"type": "number", "answer": 8000000},
        59: {"type": "number", "answer": 5000000},
    }
    agent, q_obj, calls = _make_vote_agent(monkeypatch, tmp_path, answers)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred == {"id": "q_vote", "type": "number", "answer": 5000000}
    assert details["decision_reason"] == "majority_vote_p1_p3"
    assert sorted(calls) == [42, 59, 100]


def test_tiebreak_three_way_disagreement_abstains(monkeypatch, tmp_path):
    answers = {
        42: {"type": "number", "answer": 5000000},
        100: {"type": "number", "answer": 8000000},
        59: {"type": "number", "answer": 6000000},
    }
    agent, q_obj, _ = _make_vote_agent(monkeypatch, tmp_path, answers)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"
    assert details["decision_reason"] == "consensus_disagreement_abstain"


def test_tiebreak_not_called_on_consensus(monkeypatch, tmp_path):
    answers = {
        42: {"type": "number", "answer": 5000000},
        100: {"type": "number", "answer": 5000000},
        59: {"type": "number", "answer": 1},
    }
    agent, q_obj, calls = _make_vote_agent(monkeypatch, tmp_path, answers)
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["answer"] == 5000000 and details["decision_reason"] == "consensus_match"
    assert 59 not in calls


def test_top_k_truncation_before_consensus(monkeypatch, tmp_path):
    answers = {
        42: {"type": "list", "answer": ["A", "B", "C"]},
        100: {"type": "list", "answer": ["A", "B", "C", "D"]},
        59: {"type": "list", "answer": ["X"]},
    }
    agent, q_obj, _ = _make_vote_agent(monkeypatch, tmp_path, answers, question="Top 3 món bán chạy nhất tháng này?")
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["answer"] == ["A", "B", "C"]
    assert details["decision_reason"] == "consensus_match"


def test_walk_in_marker_is_exact_match_not_substring(monkeypatch, tmp_path):
    # "Lemon tea" chứa chuỗi con "le" nhưng KHÔNG phải nhãn khách lẻ -> không được bỏ qua đồng thuận
    answers = {
        42: {"type": "string", "answer": "Trà sữa"},
        100: {"type": "string", "answer": "Lemon tea"},
        59: {"type": "string", "answer": "Cà phê"},
    }
    agent, q_obj, _ = _make_vote_agent(monkeypatch, tmp_path, answers, question="Món nào bán chạy nhất?")
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"


def test_extract_top_k():
    assert DataMateAgent._extract_top_k("Top 5 khách mua nhiều nhất") == 5
    assert DataMateAgent._extract_top_k("3 món bán chạy nhất tuần rồi là gì?") == 3
    assert DataMateAgent._extract_top_k("Doanh thu 7 ngày gần nhất") is None


# ---------------------------------------------------------------------------
# Tối ưu thời gian chạy: prefix caching, stop sequence, không gọi LLM thừa
# ---------------------------------------------------------------------------

def _capture_agent(tmp_path, monkeypatch, rulebook_text="# R\n## 1. Đơn hàng và doanh thu\nLuật A.\n", **kw):
    rb = tmp_path / "rb.md"
    rb.write_text(rulebook_text, encoding="utf-8")
    captured = []

    class CaptureClient:
        backend = "mock"
        model_name = "mock"

        def generate(self, messages, seed=42):
            captured.append(messages)
            return "```python\nresult = {'type': 'number', 'answer': 7}\n```"

    agent = DataMateAgent(data_root=tmp_path, calendar_path=tmp_path / "cal.csv", rulebook_path=rb, llm_client=CaptureClient(), **kw)
    monkeypatch.setattr(agent.store, "get_readme", lambda bid: "")
    monkeypatch.setattr(agent.store, "load_tables", lambda bid: {})
    monkeypatch.setattr(agent.store, "build_compact_schema_prompt", lambda bid, question="": "SCHEMA_SHOP")
    monkeypatch.setattr(agent.sandbox, "execute", lambda code, **kw: {"ok": True, "result": {"type": "number", "answer": 7}})
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (False, ""))
    return agent, captured


def test_system_prompt_identical_across_questions_and_contains_rulebook(monkeypatch, tmp_path):
    agent, captured = _capture_agent(tmp_path, monkeypatch)
    for i, (q, d) in enumerate([("Doanh thu hôm nay?", "2026-05-15"), ("Doanh thu tuần này?", "2026-06-20")]):
        agent.answer_question_dual({"id": f"q{i}", "business_id": "b", "as_of_date": d, "question": q})
    systems = {m[0]["content"] for m in captured}
    assert len(systems) == 1
    system = systems.pop()
    assert "Luật A." in system and "2026-05-15" not in system and "2026-06-20" not in system
    # Phần đầu user prompt (schema) dùng chung; vai trò P1/P2 nằm cuối
    users = [m[1]["content"] for m in captured]
    assert all(u.startswith("SCHEMA_SHOP") for u in users)
    assert users[0].endswith("KHÔNG viết bất kỳ chú thích nào.") and users[1].endswith("KHÔNG viết bất kỳ chú thích nào.")


def test_question_rewrite_off_by_default_and_unanswerable_skips_llm(monkeypatch, tmp_path):
    agent, captured = _capture_agent(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(agent.vietnamese_normalizer, "canonicalize", lambda **kw: calls.append(1) or "x")
    agent.answer_question_dual({"id": "q", "business_id": "b", "as_of_date": "2026-05-15", "question": "Doanh thu?"})
    assert calls == [] and len(captured) == 2
    monkeypatch.setattr(agent.rule_engine, "detect_unanswerable_deterministic", lambda **kw: (True, "ngoài dữ liệu"))
    agent.rewrite_question = True
    captured.clear()
    pred, _ = agent.answer_question_dual({"id": "q2", "business_id": "b", "as_of_date": "2026-05-15", "question": "Thời tiết?"})
    assert pred["type"] == "unanswerable" and calls == [] and captured == []


def test_openai_payload_has_closing_fence_stop(monkeypatch):
    from model.baseline import CODE_STOP_SEQUENCES
    client = QwenCoderClient(backend="openai")
    captured = {}

    class R:
        status_code = 200
        ok = True

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    def post(url, json=None, headers=None, timeout=None):
        captured.update(json)
        return R()

    monkeypatch.setattr(client._session, "post", post)
    client.generate([{"role": "user", "content": "hi"}], seed=1, stop=CODE_STOP_SEQUENCES)
    assert captured["stop"] == ["\n```\n"]
    assert "```python".find(CODE_STOP_SEQUENCES[0]) == -1


def test_submission_written_per_question_and_survives_interruption(monkeypatch, tmp_path):
    import json
    import sys
    import run_pipeline

    qs = tmp_path / "q.jsonl"
    qs.write_text("\n".join(json.dumps({"id": f"q{i}", "business_id": "b", "as_of_date": "2026-01-01", "question": "x"}) for i in range(3)), encoding="utf-8")
    out = tmp_path / "result" / "submission.jsonl"

    def fake_answer(self, q_obj):
        if q_obj["id"] == "q2":
            raise KeyboardInterrupt()
        return {"id": q_obj["id"], "type": "number", "answer": 1}, {"id": q_obj["id"], "is_consensus": True}

    monkeypatch.setattr(DataMateAgent, "answer_question_dual", fake_answer)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--questions", str(qs), "--data-root", str(tmp_path), "--backend", "mock",
        "--output", str(out), "--concurrency", "1", "--allow-default-rulebook",
    ])
    class _Store:
        def load_tables(self, bid, use_cache=True):
            return {}

    def fake_init(self, *a, **k):
        self.store = _Store()

    monkeypatch.setattr(DataMateAgent, "__init__", fake_init)
    import pytest
    with pytest.raises(KeyboardInterrupt):
        run_pipeline.main()
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [l["id"] for l in lines] == ["q0", "q1"]


def test_rulebook_discovered_next_to_data_and_missing_is_error(monkeypatch, tmp_path):
    import json
    import sys
    import pytest
    import run_pipeline

    split = tmp_path / "split"
    (split / "data").mkdir(parents=True)
    (split / "questions.jsonl").write_text(json.dumps({"id": "q", "business_id": "b", "as_of_date": "2026-01-01", "question": "x"}), encoding="utf-8")
    argv = ["run_pipeline.py", "--questions", str(split / "questions.jsonl"), "--data-root", str(split / "data"),
            "--backend", "mock", "--output", str(tmp_path / "o.jsonl")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        run_pipeline.main()
    assert "rulebook" in str(e.value).lower()

    (split / "rulebook.md").write_text("# Rulebook của tập này\n", encoding="utf-8")
    seen = {}

    class _Stop(Exception):
        pass

    def fake_init(self, data_root, calendar_path, rulebook_path, *a, **k):
        seen["rb"], seen["cal"] = rulebook_path, calendar_path
        raise _Stop()

    monkeypatch.setattr(DataMateAgent, "__init__", fake_init)
    with pytest.raises(_Stop):
        run_pipeline.main()
    assert seen["rb"] == split / "rulebook.md"


def test_parallel_ingest_populates_cache_and_matches_sequential(tmp_path):
    import pandas as pd
    import run_pipeline
    from src.ingest import BusinessDataStore

    root = tmp_path / "data"
    for b in ("b1", "b2", "b3"):
        d = root / b
        d.mkdir(parents=True)
        pd.DataFrame({"Mã đơn": [f"{b}-{i}" for i in range(5)], "Ngày": ["2026-01-01"] * 5, "Thành tiền": [10, 20, 30, 40, 50]}).to_csv(d / "ban.csv", index=False)
    cache = tmp_path / "cache"
    store = BusinessDataStore(root, cache_root=cache)
    run_pipeline._parallel_ingest(store, ["b1", "b2", "b3"], workers=2)
    assert all((cache / b / "_manifest.json").exists() for b in ("b1", "b2", "b3"))
    seq = BusinessDataStore(root, cache_root=tmp_path / "cache_seq").load_tables("b2")
    par = BusinessDataStore(root, cache_root=cache).load_tables("b2")
    assert list(seq) == list(par)
    assert seq["ban"]["Thành tiền"].tolist() == par["ban"]["Thành tiền"].tolist()
    # workers=1 hoặc không có shop chờ -> không làm gì, không lỗi
    run_pipeline._parallel_ingest(store, ["b1"], workers=1)
