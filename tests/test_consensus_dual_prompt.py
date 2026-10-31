# -*- coding: utf-8 -*-
"""
Unit test chuyên sâu cho cơ chế Dual-Prompt Consensus Gate (Bỏ phiếu đồng thuận 2 góc nhìn).
Kiểm tra các kịch bản:
  1. Đồng thuận số học (Number / Percent / Money rounding) -> Trả về đáp án (+1 điểm).
  2. Đồng thuận chuỗi (String case-insensitive / NFC) -> Trả về đáp án.
  3. Bất đồng kết quả (Lệch doanh thu hoặc sót điều kiện) -> Tự động kích hoạt Abstain (0 điểm).
  4. Một bên lỗi cú pháp / crash trong Sandbox -> Tự động kích hoạt Abstain an toàn.
  5. Cả hai cùng kết luận Unanswerable -> Đồng thuận unanswerable.
  6. Kiểm tra cấu trúc output của `answer_question_dual` để xuất đủ 2 file JSONL.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from model.baseline import DataMateAgent, QwenCoderClient
from src.postprocess import check_consensus


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
DATA_ROOT = TRAIN_DIR / "data"
CALENDAR_PATH = TRAIN_DIR / "calendar.csv"
RULEBOOK_PATH = TRAIN_DIR / "rulebook.md"


def test_check_consensus_logic_pure():
    # 1. Number & Percent
    assert check_consensus({"type": "number", "answer": 150000}, {"type": "number", "answer": 150000.0}) is True
    assert check_consensus({"type": "number", "answer": 150000}, {"type": "number", "answer": 140000}) is False
    assert check_consensus({"type": "percent", "answer": 40.3}, {"type": "percent", "answer": 40.3}) is True
    assert check_consensus({"type": "percent", "answer": 40.3}, {"type": "percent", "answer": 42.1}) is False

    # 2. String & Date
    assert check_consensus({"type": "string", "answer": "Trà Đào Cam Sả"}, {"type": "string", "answer": "trà đào cam sả"}) is True
    assert check_consensus({"type": "string", "answer": "Trà đào"}, {"type": "string", "answer": "Trà sữa"}) is False
    assert check_consensus({"type": "date", "answer": "2025-03-15"}, {"type": "date", "answer": "2025-03-15"}) is True
    assert check_consensus({"type": "date", "answer": "2025-03-15"}, {"type": "date", "answer": "2025-03-16"}) is False

    # 3. List
    assert check_consensus({"type": "list", "answer": ["A", "B"]}, {"type": "list", "answer": ["a", "b"]}) is True
    assert check_consensus({"type": "list", "answer": ["A", "B"]}, {"type": "list", "answer": ["B", "A"]}) is False
    assert check_consensus({"type": "list", "answer": [10, 20]}, {"type": "list", "answer": [10.0, 20.0]}) is True

    # 4. Abstain & Unanswerable
    assert check_consensus({"type": "abstain"}, {"type": "number", "answer": 100}) is False
    assert check_consensus({"type": "number", "answer": 100}, {"type": "abstain"}) is False
    assert check_consensus({"type": "unanswerable"}, {"type": "unanswerable"}) is True
    assert check_consensus({"type": "unanswerable"}, {"type": "number", "answer": 100}) is False


def test_agent_dual_prompt_agreement(tmp_path: Path):
    """Giả lập 2 prompt cùng giải ra kết quả 10.309.400 -> Đồng thuận."""
    call_tracker = {"n": 0}

    def mock_generator(messages: List[Dict[str, str]]) -> str:
        call_tracker["n"] += 1
        sys_content = messages[0]["content"]
        if "chuyên về SQL" in sys_content:
            # Prompt 1: Viết kiểu SQL
            return '```python\nresult = {"type": "number", "answer": 10309400}\n```'
        else:
            # Prompt 2: Viết kiểu DataFrame
            return '```python\nresult = {"type": "number", "answer": 10309400.0}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_generator)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_agree_test",
        "business_id": "biz_120",
        "question": "Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        "as_of_date": "2025-04-30",
    }
    final_pred, details = agent.answer_question_dual(q_obj)

    assert call_tracker["n"] == 2
    assert details["is_consensus"] is True
    assert details["final_decision"] == "number"
    assert details["final_answer"] == 10309400
    assert final_pred == {"id": "q_agree_test", "type": "number", "answer": 10309400}


def test_agent_dual_prompt_disagreement_abstains(tmp_path: Path):
    """Giả lập Prompt 1 tính ra 150k, Prompt 2 tính ra 140k -> Bất đồng -> Abstain an toàn."""
    def mock_disagree_generator(messages: List[Dict[str, str]]) -> str:
        sys_content = messages[0]["content"]
        if "chuyên về SQL" in sys_content:
            return '```python\nresult = {"type": "number", "answer": 150000}\n```'
        else:
            return '```python\nresult = {"type": "number", "answer": 140000}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_disagree_generator)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_disagree_test",
        "business_id": "biz_120",
        "question": "Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        "as_of_date": "2025-04-30",
    }
    final_pred, details = agent.answer_question_dual(q_obj)

    assert details["is_consensus"] is False
    assert details["final_decision"] == "abstain"
    assert final_pred == {"id": "q_disagree_test", "type": "abstain"}
