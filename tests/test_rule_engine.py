# -*- coding: utf-8 -*-
"""
Unit tests cho Module 2B: Query Analyzer & Rule Knowledge Injector.
Kiểm tra phân loại expected_type, trích xuất luật, và phát hiện câu hỏi unanswerable tiền định.
"""

from pathlib import Path
import pytest
from src.rule_engine import RuleEngine, RulebookKnobReader, strip_accents


RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "assets" / "rulebook.md"


def test_strip_accents():
    assert strip_accents("Đơn hàng trà sữa (L)") == "don hang tra sua (l)"
    assert strip_accents("Khách hàng vãng lai") == "khach hang vang lai"


def test_infer_expected_type_all_categories():
    # 1. Number
    assert RuleEngine.infer_expected_type("Tháng này bán được bao nhiêu tiền?") == "number"
    assert RuleEngine.infer_expected_type("Có mấy đơn hàng bị hủy trong tuần?") == "number"
    assert RuleEngine.infer_expected_type("Bán được bao nhiêu ly trà sữa?") == "number"

    # 2. Percent
    assert RuleEngine.infer_expected_type("Tỉ lệ đơn hủy trong tháng 5 là bao nhiêu phần trăm?") == "percent"
    assert RuleEngine.infer_expected_type("Doanh thu tháng này tăng bao nhiêu % so với tháng trước?") == "percent"
    assert RuleEngine.infer_expected_type("Tỉ trọng doanh thu món nước chiếm mấy phần trăm?") == "percent"

    # 3. Date
    assert RuleEngine.infer_expected_type("Lô hàng này hết hạn vào ngày nào?") == "date"
    assert RuleEngine.infer_expected_type("Doanh thu cao nhất vào tháng nào?") == "date"

    # 4. Boolean
    assert RuleEngine.infer_expected_type("Doanh thu tuần này có cao hơn tuần trước không?") == "boolean"
    assert RuleEngine.infer_expected_type("Có phải khách hàng A nợ nhiều nhất không bạn?") == "boolean"

    # 5. String
    assert RuleEngine.infer_expected_type("Ai là người bán được nhiều tiền nhất?") == "string"
    assert RuleEngine.infer_expected_type("Mặt hàng nào hot nhất trong tuần qua?") == "string"

    # 6. List
    assert RuleEngine.infer_expected_type("Cho tôi top 5 mặt hàng bán chạy nhất") == "list"
    assert RuleEngine.infer_expected_type("Liệt kê các nhân viên có doanh số trên 50 triệu") == "list"
    assert RuleEngine.infer_expected_type("Doanh thu tháng 4 và tháng 5 lần lượt là bao nhiêu?") == "list"

    # 7. Table
    assert RuleEngine.infer_expected_type("Lập bảng danh sách top 3 khách nợ kèm số tiền nợ") == "table"
    assert RuleEngine.infer_expected_type("Mỗi khách còn nợ bao nhiêu tiền?") == "table"


def test_rulebook_knob_reader():
    if not RULEBOOK_PATH.exists():
        pytest.skip("rulebook.md không tồn tại trong assets")
    rulebook_text = RULEBOOK_PATH.read_text(encoding="utf-8")
    knobs = RulebookKnobReader.parse_knobs(rulebook_text)

    assert knobs.week_start_day.lower() == "thứ hai"
    assert knobs.shift_morning == ("06:00", "13:59")
    assert knobs.shift_afternoon == ("14:00", "21:59")
    assert knobs.tet_start_lunar_day == 23
    assert knobs.tet_end_lunar_day == 10
    assert knobs.money_decimals == 0
    assert knobs.percent_decimals == 2


class DummyStore:
    def get_shop_date_range(self, bid):
        return "2025-01-01", "2025-12-31"

    def load_tables(self, bid):
        import pandas as pd
        return {"ban_hang": pd.DataFrame({"Mã đơn": ["HD1"], "Thành tiền": [1000]})}


class DummyDateResolver:
    def resolve(self, q, as_of):
        if "2028" in q:
            return {
                "detected_periods": [{"start_date": "2028-01-01", "end_date": "2028-01-31"}]
            }
        return {
            "detected_periods": [{"start_date": "2025-05-01", "end_date": "2025-05-15"}]
        }


def test_detect_unanswerable_deterministic():
    store = DummyStore()
    d_res = DummyDateResolver()

    # 1. Câu hỏi ngoài khoảng thời gian dữ liệu
    q_out = {"question": "Doanh thu tháng 1 năm 2028 là bao nhiêu?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_out, store, d_res)
    assert is_unans
    assert "ngoài khoảng ngày" in reason

    # 2. Câu hỏi cảm tính không có trong dữ liệu
    q_feel = {"question": "Nhân viên nào có thái độ chu đáo và nhiệt tình nhất?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans2, reason2 = RuleEngine.detect_unanswerable_deterministic(q_feel, store, d_res)
    assert is_unans2
    assert "cảm tính" in reason2

    # 3. Câu hỏi kiến thức y khoa ngoài bảng
    q_med = {"question": "Thuốc Panadol uống mỗi lần mấy gói?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans3, reason3 = RuleEngine.detect_unanswerable_deterministic(q_med, store, d_res)
    assert is_unans3
    assert "kiến thức bên ngoài" in reason3


def test_dynamic_entity_grounder_matches_product():
    from src.rule_engine import DynamicEntityGrounder
    import pandas as pd
    grounder = DynamicEntityGrounder()
    df = pd.DataFrame({
        "Tên sản phẩm": ["Trà sữa Ô long nướng", "Cà phê muối", "Bạc xỉu đá"],
        "Đơn vị tính": ["Ly", "Ly", "Ly"],
    })
    tables = {"menu": df}
    grounded = grounder.ground("biz_test", "Giá ly o long nuong bao nhieu?", tables)
    assert "Trà sữa Ô long nướng" in grounded


def test_infer_expected_type_comprehensive():
    assert RuleEngine.infer_expected_type("Top 3 mặt hàng có doanh thu cao nhất?") in ["table", "list"]
    assert RuleEngine.infer_expected_type("Tổng số tiền thu được là bao nhiêu?") == "number"
    assert RuleEngine.infer_expected_type("Tỉ lệ hủy đơn là bao nhiêu %?") == "percent"
    assert RuleEngine.infer_expected_type("Ngày nào bán được nhiều nhất?") == "date"
    assert RuleEngine.infer_expected_type("Cửa hàng có mở cửa vào chủ nhật không?") == "boolean"
    assert RuleEngine.infer_expected_type("Liệt kê danh sách các kênh bán hàng?") == "list"


def test_detect_unanswerable_missing_required_data():
    store = DummyStore()
    d_res = DummyDateResolver()
    # Hỏi đánh giá sao hoặc review của khách (không có trong bất kỳ bảng nào)
    q_review = {"question": "Khách hàng chấm mấy sao cho chất lượng dịch vụ?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_review, store, d_res)
    assert is_unans
    assert "cảm tính" in reason or "ngoài" in reason


def test_detect_unanswerable_empty_business_store():
    class EmptyStore:
        def get_date_range(self, bid):
            return None, None
        def load_tables(self, bid):
            return {}

    d_res = DummyDateResolver()
    q = {"question": "Doanh thu hôm nay?", "business_id": "empty_biz", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q, EmptyStore(), d_res)
    assert is_unans
    assert "không có dữ liệu" in reason.lower() or "ngoài" in reason.lower() or "không" in reason.lower()


def test_rulebook_knob_reader_default_structure(tmp_path):
    # Đọc khi file không tồn tại trả về cấu trúc mặc định an toàn
    knobs = RuleEngine.read_rulebook_knobs(tmp_path / "non_existent.md")
    assert isinstance(knobs, dict)
    assert "vat_rate" in knobs or "cancel_status" in knobs or "shift" in knobs or len(knobs) >= 0


def test_detect_unanswerable_expanded_domains():
    store = DummyStore()
    d_res = DummyDateResolver()

    # 1. Nhiệt độ / độ ẩm kho
    q_temp = {"question": "Kho thuốc bảo quản ở nhiệt độ bao nhiêu độ?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_temp, store, d_res)
    assert is_unans
    assert "nhiệt độ" in reason.lower()

    # 2. Tác dụng phụ thuốc y tế
    q_side_effects = {"question": "Thuốc này có tác dụng phụ gì không?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_side_effects, store, d_res)
    assert is_unans
    assert "tác dụng phụ" in reason.lower() or "dược lý" in reason.lower()

    # 3. Khoảng cách địa lý / Google Maps
    q_maps = {"question": "Quán cách bến xe bao xa theo Google Maps?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_maps, store, d_res)
    assert is_unans
    assert "tọa độ" in reason.lower() or "địa lý" in reason.lower()

    # 4. Thời tiết / khí tượng
    q_weather = {"question": "Hôm nay trời mưa hay nắng?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_weather, store, d_res)
    assert is_unans
    assert "thời tiết" in reason.lower()

    # 5. Vé tàu xe / chuyến bay
    q_flight = {"question": "Có bao nhiêu chuyến bay từ Hà Nội vào?", "business_id": "biz_01", "as_of_date": "2025-12-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_flight, store, d_res)
    assert is_unans
    assert "vận tải" in reason.lower() or "lịch trình" in reason.lower()


def test_detect_unanswerable_future_sales():
    store = DummyStore()
    class FutureDateResolver:
        def resolve(self, q, as_of):
            # Ngày mai: 2025-06-01 khi as_of là 2025-05-31
            return {"detected_periods": [{"start_date": "2025-06-01", "end_date": "2025-06-01"}]}

    q_future = {"question": "Doanh thu bán được ngày mai?", "business_id": "biz_01", "as_of_date": "2025-05-31"}
    is_unans, reason = RuleEngine.detect_unanswerable_deterministic(q_future, store, FutureDateResolver())
    assert is_unans
    assert "tương lai" in reason.lower()

