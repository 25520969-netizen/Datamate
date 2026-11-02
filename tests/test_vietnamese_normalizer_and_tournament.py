# -*- coding: utf-8 -*-
"""
Unit tests cho:
1. Module Vietnamese Context-Aware Canonicalizer (`src/vietnamese_normalizer.py`):
   - Chuẩn hoá Telex typos & Unicode NFC
   - Canonicalize bảo toàn thực thể, số, ngày tháng
2. Cơ chế Agentar-Scale-SQL Tournament Selection (`model/baseline.py`):
   - Chế độ tournament: cứu điểm khi P1 và P2 ra 2 số hợp lệ khác nhau thay vì ép Abstain
   - Chế độ strict: vẫn giữ tính tương thích ngược
3. Kiểm tra an toàn tài nguyên: RAM tiến trình luôn sâu dưới 8GB
"""

import os
from pathlib import Path
from typing import Dict, List

import pytest

from model.baseline import DataMateAgent, QwenCoderClient
from src.vietnamese_normalizer import VietnameseNormalizer, to_nfc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
DATA_ROOT = TRAIN_DIR / "data"
CALENDAR_PATH = TRAIN_DIR / "calendar.csv"
RULEBOOK_PATH = TRAIN_DIR / "rulebook.md"


def test_vietnamese_normalizer_telex_and_nfc():
    """Kiểm tra lớp deterministic sửa lỗi gõ Telex và chuẩn hóa NFC."""
    raw = "hqua cos tổng cộng bao nhieu ddown banf giao cho khach hangf"
    cleaned = VietnameseNormalizer.clean_deterministic(raw)
    assert "có" in cleaned
    assert "đơn" in cleaned
    assert "hàng" in cleaned
    assert "ddown" not in cleaned
    assert "cos" not in cleaned


def test_vietnamese_normalizer_canonicalize_mock():
    """Kiểm tra canonicalize với mock LLM hoạt động chuẩn theo DART-SQL."""
    captured_messages = []

    class MockClient:
        backend = "mock"
        _supports_canonicalize = True

        def generate(self, messages, seed=42):
            captured_messages.append(messages)
            return "Doanh thu tháng 1 năm 2026 là bao nhiêu?"

    norm = VietnameseNormalizer(llm_client=MockClient())
    schema = 'CREATE TABLE "ban_hang" ("Ngày" DATE, "revenue_norm" DOUBLE);'
    res = norm.canonicalize("dthu th1 2026 bnhiu z nhi", schema_summary=schema, as_of_date="2026-02-01")

    assert res == "Doanh thu tháng 1 năm 2026 là bao nhiêu?"
    assert len(captured_messages) == 1
    assert "DART-SQL" in VietnameseNormalizer.__doc__ or "canonical" in captured_messages[0][0]["content"].lower()


def test_agent_tournament_selection_rescues_abstain(tmp_path: Path):
    """
    Kiểm tra cơ chế Tournament Selection (Agentar-Scale-SQL):
    Khi P1 tính ra 150.000đ và P2 tính ra 148.000đ:
    - Ở chế độ 'tournament': chấp nhận P1 (Primary SQL Engine), ghi nhận điểm thay vì mất trắng 0 điểm.
    """
    def mock_disagree_gen(messages: List[Dict[str, str]]) -> str:
        sys_c = messages[0]["content"]
        if "chuyên về SQL" in sys_c:
            return '```python\nresult = {"type": "number", "answer": 150000}\n```'
        else:
            return '```python\nresult = {"type": "number", "answer": 148000}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_disagree_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
        arbitration_strategy="tournament",
    )

    q_obj = {
        "id": "q_tournament_test",
        "business_id": "biz_120",
        "question": "Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        "as_of_date": "2025-04-30",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert pred["type"] == "number"
    assert pred["answer"] == 150000
    assert details["decision_reason"] == "tournament_selection_p1_sql"
    assert details["final_decision"] == "number"


def test_memory_consumption_safety():
    """Kiểm tra mức tiêu thụ RAM tiến trình luôn an toàn dưới 2GB (cách xa giới hạn 8GB)."""
    import resource
    import sys
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        mb = usage / (1024 * 1024)  # macOS ru_maxrss is in bytes
    else:
        mb = usage / 1024  # Linux ru_maxrss is in KB
    print(f"\n[RAM Check] Current Process Max RSS: {mb:.2f} MB")
    assert mb < 2048.0, f"RAM usage ({mb:.2f} MB) vượt ngưỡng an toàn 2GB!"
