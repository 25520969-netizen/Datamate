# -*- coding: utf-8 -*-
"""
Bộ kiểm thử tự động xác nhận tính tối ưu, tính tổng quát và khả năng tự phục hồi
sau khi tối ưu hóa QwenCoderClient, check_consensus, và Tiered Arbitration.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from model.baseline import DataMateAgent, QwenCoderClient
from src.postprocess import check_consensus


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_qwen_coder_client_url_normalization():
    """Kiểm tra QwenCoderClient chuẩn hóa đúng mọi biến thể URL api_base."""
    variants = [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/chat/completions", "http://localhost:8000/v1/chat/completions"),
    ]
    for raw_url, expected in variants:
        client = QwenCoderClient(backend="openai", api_base=raw_url)
        assert client._get_openai_url() == expected, f"Failed for {raw_url}"


def test_check_consensus_list_and_tables_strict_order():
    """Kiểm tra check_consensus yêu cầu thứ tự chính xác cho list (như Top N) kèm chuẩn hóa NFC."""
    # 1. Danh sách cùng thứ tự, khác hoa thường / NFC
    l1 = {"type": "list", "answer": ["Trà đào cam sả", "Cà phê sữa đá"]}
    l2 = {"type": "list", "answer": ["trà đào cam sả", "cà phê sữa đá"]}
    assert check_consensus(l1, l2) is True

    # Khác thứ tự trong xếp hạng -> False
    l_reverse = {"type": "list", "answer": ["Cà phê sữa đá", "Trà đào cam sả"]}
    assert check_consensus(l1, l_reverse) is False

    # 2. Bảng đúng thứ tự hàng
    t1 = {
        "type": "table",
        "answer": {
            "columns": ["Khách hàng", "Doanh thu"],
            "rows": [["Nguyễn Văn A", 5000000], ["Trần Thị B", 8000000]],
        },
    }
    t2 = {
        "type": "table",
        "answer": {
            "columns": ["Khách hàng", "Doanh thu"],
            "rows": [["nguyễn văn a", 5000000.0], ["trần thị b", 8000000.0]],
        },
    }
    assert check_consensus(t1, t2) is True

    # Bảng sai số tiền -> False
    t3 = {
        "type": "table",
        "answer": {
            "columns": ["Khách hàng", "Doanh thu"],
            "rows": [["Nguyễn Văn A", 5000000], ["Trần Thị B", 9999999]],
        },
    }
    assert check_consensus(t1, t3) is False


def test_check_consensus_financial_relative_precision():
    """Kiểm tra sai số tương đối cho phép với số tài chính lớn hàng tỷ đồng."""
    p1 = {"type": "number", "answer": 1500000000.0}
    p2 = {"type": "number", "answer": 1500000005.0}  # Lệch 5 đồng trên 1.5 tỷ (tỷ lệ 3e-9 < 1e-5)
    assert check_consensus(p1, p2) is True

    # Lệch lớn (1.5 tỷ vs 1.6 tỷ) -> False
    p3 = {"type": "number", "answer": 1600000000.0}
    assert check_consensus(p1, p3) is False


def test_agent_tiered_arbitration_adaptive_fallback(tmp_path: Path):
    """
    Kiểm tra cơ chế Adaptive Arbitration:
    - Khi Prompt 1 thành công nhưng Prompt 2 crash trong Sandbox:
      Chấp nhận đáp án Prompt 1 thay vì vứt bỏ biến thành abstain.
    """
    data_root = PROJECT_ROOT / "train_preprocessed_full"
    if not data_root.exists():
        data_root = PROJECT_ROOT / "train" / "data"
    calendar_path = PROJECT_ROOT / "train" / "calendar.csv"
    rulebook_path = PROJECT_ROOT / "train" / "rulebook.md"

    # Giả lập: Lượt gọi Prompt 1 trả về code đúng, Prompt 2 trả về code lỗi cú pháp
    call_idx = {"count": 0}

    def mock_generator(messages: List[Dict[str, str]]) -> str:
        call_idx["count"] += 1
        sys_msg = messages[0]["content"]
        if "SQL" in sys_msg:
            # Code SQL chuẩn trên bảng don_hang của biz_127
            return """```python
df = tables["don_hang"]
result = {"type": "number", "answer": 12345000}
```"""
        else:
            # Prompt 2 luôn crash trong sandbox để kiểm thử fallback
            return "```python\nraise RuntimeError('Prompt 2 sandbox crash')\n```"

    client = QwenCoderClient(backend="custom", custom_generator=mock_generator)
    agent = DataMateAgent(
        data_root=data_root,
        calendar_path=calendar_path,
        rulebook_path=rulebook_path,
        llm_client=client,
        consensus_mode="dual",
        arbitration_strategy="adaptive",
    )

    q_obj = {
        "id": "q_test_adaptive",
        "business_id": "biz_127",
        "question": "Tổng doanh thu năm 2026 là bao nhiêu?",
        "as_of_date": "2026-07-28",
    }
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "number"
    assert pred["answer"] == 12345000
    assert details["decision_reason"] == "p1_valid_p2_failed_fallback"


def test_agent_disagreement_abstains_to_protect_score(tmp_path: Path):
    """
    Kiểm tra: Khi 2 Prompt sinh 2 con số khác nhau (bất đồng logic thực sự),
    Agent bắt buộc phải ABSTAIN để tránh bị trừ 1 điểm (-1).
    """
    data_root = PROJECT_ROOT / "train_preprocessed_full"
    if not data_root.exists():
        data_root = PROJECT_ROOT / "train" / "data"
    calendar_path = PROJECT_ROOT / "train" / "calendar.csv"
    rulebook_path = PROJECT_ROOT / "train" / "rulebook.md"

    call_idx = {"count": 0}

    def mock_generator(messages: List[Dict[str, str]]) -> str:
        call_idx["count"] += 1
        sys_msg = messages[0]["content"]
        if "SQL" in sys_msg or call_idx["count"] % 2 == 1:
            return '```python\nresult = {"type": "number", "answer": 100000}\n```'
        else:
            return '```python\nresult = {"type": "number", "answer": 200000}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_generator)
    agent = DataMateAgent(
        data_root=data_root,
        calendar_path=calendar_path,
        rulebook_path=rulebook_path,
        llm_client=client,
        consensus_mode="dual",
        arbitration_strategy="adaptive",
    )

    q_obj = {
        "id": "q_test_disagree",
        "business_id": "biz_127",
        "question": "Doanh thu ngày hôm nay?",
        "as_of_date": "2026-07-28",
    }
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"
    assert details["decision_reason"] == "consensus_disagreement_abstain"
    assert details["is_consensus"] is False
