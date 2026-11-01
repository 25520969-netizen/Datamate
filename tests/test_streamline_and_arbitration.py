# -*- coding: utf-8 -*-
"""
Unit tests cho:
  1. Tinh gọn và Khử nhiễu Prompt (De-noising Schema & Date Prompt).
  2. Chuẩn hóa bộ gõ đầu vào (Telex Input Sanitization).
  3. Cơ chế Trọng tài Đặc trưng Truy vấn (Query-Feature Arbitration):
     - Khách định danh vs Khách lẻ (Retail Customer Arbitration)
     - Tỉ lệ hủy đơn thực tế vs Lọc ngược 0.0% (Cancellation Rate Arbitration)
     - Doanh thu bảng Master vs Detail Bloat do JOIN nhân dòng (Master-Detail Bloat Arbitration)
     - Ưu tiên Kiểu phần trăm khi câu hỏi yêu cầu % (Type Priority Arbitration)
"""

from pathlib import Path
from typing import Dict, List
import pytest

from model.baseline import DataMateAgent, QwenCoderClient
from src.date_resolver import DateResolver
from src.ingest import BusinessDataStore
from src.rule_engine import RuleEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
DATA_ROOT = TRAIN_DIR / "data"
CALENDAR_PATH = TRAIN_DIR / "calendar.csv"
RULEBOOK_PATH = TRAIN_DIR / "rulebook.md"


def test_date_resolver_prompt_denoising():
    """Kiểm tra DateResolver khử nhiễu khi câu hỏi đã có mốc cụ thể."""
    resolver = DateResolver(calendar_path=CALENDAR_PATH)

    # 1. Câu hỏi ngày cụ thể: không chứa các từ so sánh -> chỉ in mốc mục tiêu và as_of_date
    prompt_specific = resolver.format_for_prompt("Cho mình bảng 3 khách còn nợ đến hết ngày 05/08/2026", "2026-08-05")
    assert "MỐC THỜI GIAN MỤC TIÊU CẦN TRUY VẤN" in prompt_specific
    assert "2026-08-05" in prompt_specific
    assert "Hôm nay (`as_of_date`): 2026-08-05" in prompt_specific
    assert "Tuần trước" not in prompt_specific
    assert "Từ đầu năm tới nay" not in prompt_specific

    # 2. Câu hỏi so sánh tăng trưởng: cần in đủ các mốc đối chiếu
    prompt_comp = resolver.format_for_prompt("Doanh thu tháng này so với tháng trước tăng bao nhiêu %", "2026-08-05")
    assert "Hôm nay (`as_of_date`): 2026-08-05" in prompt_comp
    assert "Tuần trước" in prompt_comp
    assert "Tháng trước" in prompt_comp


def test_schema_pruning_biz_123():
    """Kiểm tra Schema Pruning giảm 50%+ kích thước prompt trên shop có 17 bảng."""
    store = BusinessDataStore(data_root=DATA_ROOT)
    prompt = store.build_compact_schema_prompt("biz_123", question="hôm qua bán được bao nhiêu đơn")

    # Bản cũ dài 9002 ký tự với 17 bảng; bản mới phải dưới 5000 ký tự
    assert len(prompt) < 5500
    assert 'CREATE TABLE "don_hang"' in prompt
    # Các bảng không liên quan bị ẩn khỏi chi tiết CREATE TABLE
    assert 'CREATE TABLE "nhan_vien"' not in prompt
    assert 'CREATE TABLE "nha_cung_cap"' not in prompt
    assert 'CREATE TABLE "lich_su_gia"' not in prompt
    # Vẫn có ghi chú các bảng phụ sẵn sàng trong kết nối con
    assert "Các bảng phụ khác" in prompt


def test_telex_input_sanitization(tmp_path: Path):
    """Kiểm tra câu hỏi dính lỗi gõ telex (ddown, cos, mawtj hangf) được làm sạch trước khi gửi LLM."""
    captured_questions = []

    def mock_gen(messages: List[Dict[str, str]]) -> str:
        user_msg = messages[1]["content"]
        captured_questions.append(user_msg)
        return '```python\nresult = {"type": "number", "answer": 26}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_telex_test",
        "business_id": "biz_120",
        "question": "hqua cos tổng cộng nhiêu ddown",
        "as_of_date": "2025-04-30",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert "ddown" not in captured_questions[0]
    assert "đơn" in captured_questions[0]
    assert "có tổng cộng nhiêu đơn" in captured_questions[0]
    assert pred == {"id": "q_telex_test", "type": "number", "answer": 26}


def test_retail_customer_arbitration(tmp_path: Path):
    """Kiểm tra trọng tài ưu tiên Tên khách cụ thể khi bên còn lại ra 'Khách lẻ'."""
    def mock_gen(messages: List[Dict[str, str]]) -> str:
        sys_c = messages[0]["content"]
        if "chuyên về SQL" in sys_c:
            # P1 ra tên khách thật
            return '```python\nresult = {"type": "string", "answer": "Vũ Châu Mai"}\n```'
        else:
            # P2 quên lọc khách lẻ
            return '```python\nresult = {"type": "string", "answer": "Khách lẻ"}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_cust_test",
        "business_id": "biz_120",
        "question": "Khách nào chi nhiều tiền nhất cho cửa hàng trong tháng 5/2026 nhỉ?",
        "as_of_date": "2026-05-31",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert details["decision_reason"] == "retail_customer_arbitration_p1"
    assert pred == {"id": "q_cust_test", "type": "string", "answer": "Vũ Châu Mai"}


def test_cancel_rate_arbitration(tmp_path: Path):
    """Kiểm tra trọng tài chọn tỉ lệ hủy thực tế (1.91%) thay vì 0.0% do lọc ngược."""
    def mock_gen(messages: List[Dict[str, str]]) -> str:
        sys_c = messages[0]["content"]
        if "chuyên về SQL" in sys_c:
            return '```python\nresult = {"type": "percent", "answer": 1.91}\n```'
        else:
            return '```python\nresult = {"type": "percent", "answer": 0.0}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_cancel_test",
        "business_id": "biz_120",
        "question": "Trong tháng này, tỉ lệ đơn huỷ là bao nhiêu phần trăm?",
        "as_of_date": "2026-07-28",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert "cancel_rate" in details["decision_reason"]
    assert pred == {"id": "q_cancel_test", "type": "percent", "answer": 1.91}


def test_revenue_master_vs_detail_bloat_arbitration(tmp_path: Path):
    """Kiểm tra trọng tài chọn doanh thu từ bảng Master khi P2 JOIN detail làm đội tiền gấp đôi."""
    def mock_gen(messages: List[Dict[str, str]]) -> str:
        sys_c = messages[0]["content"]
        if "chuyên về SQL" in sys_c:
            return '```python\nresult = {"type": "number", "answer": 296157500}\n```'
        else:
            return (
                '```python\n'
                '# SELECT hd.*, hdt.* FROM DanhSachHoaDon hd JOIN DanhSachHoaDonChiTiet hdt ...\n'
                'result = {"type": "number", "answer": 574484000}\n'
                '```'
            )

    client = QwenCoderClient(backend="custom", custom_generator=mock_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_rev_test",
        "business_id": "biz_120",
        "question": "Tháng 10/2025 cửa hàng bán được tổng cộng bao nhiêu tiền vậy?",
        "as_of_date": "2025-10-31",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert details["decision_reason"] == "p1_revenue_master_arbitration"
    assert pred == {"id": "q_rev_test", "type": "number", "answer": 296157500}


def test_type_priority_percent_arbitration(tmp_path: Path):
    """Kiểm tra ưu tiên đáp án dạng percent khi câu hỏi hỏi % nhưng một bên trả về number."""
    def mock_gen(messages: List[Dict[str, str]]) -> str:
        sys_c = messages[0]["content"]
        if "chuyên về SQL" in sys_c:
            return '```python\nresult = {"type": "percent", "answer": 12.59}\n```'
        else:
            return '```python\nresult = {"type": "number", "answer": 942681600}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_gen)
    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
    )

    q_obj = {
        "id": "q_pct_test",
        "business_id": "biz_120",
        "question": "Doanh thu tháng 3 năm 2026 tăng trưởng bao nhiêu % so với tháng 3 năm 2025?",
        "as_of_date": "2026-03-31",
    }
    pred, details = agent.answer_question_dual(q_obj)

    assert details["decision_reason"] == "type_priority_arbitration_p1"
    assert pred == {"id": "q_pct_test", "type": "percent", "answer": 12.59}
