# -*- coding: utf-8 -*-
"""
Hồi quy cho các lỗi tổng quát phát hiện khi rà soát (không gắn với shop cụ thể nào).
"""

from pathlib import Path

import pandas as pd

from src.date_resolver import DateResolver
from src.ingest import (
    BusinessDataStore,
    normalize_dataframe_nfc,
    normalize_order_status,
    pick_order_status_col,
)
from src.postprocess import postprocess_and_guard
from src.rule_engine import RuleEngine

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def test_paid_status_is_not_returned_order():
    assert normalize_order_status("Đã trả") == "Đã trả"
    assert normalize_order_status("Đang kiểm tra") == "Đang kiểm tra"
    assert normalize_order_status("Khách trả hàng 1 phần") == "Trả hàng"
    assert normalize_order_status("Đã huỷ") == "Đã hủy"
    assert normalize_order_status("Đang giao") == "Đang giao"


def test_order_status_column_skips_payment_status():
    df = pd.DataFrame({
        "Trạng thái thanh toán": ["Đã trả", "Chưa trả"],
        "Tình trạng đơn": ["Hoàn thành", "Đang giao"],
    })
    assert pick_order_status_col(df) == "Tình trạng đơn"


def test_month_first_date_column_is_read_consistently():
    df = normalize_dataframe_nfc(pd.DataFrame({"Ngày": ["03/04/2026", "03/15/2026"]}))
    assert df["Ngày"].tolist() == ["2026-03-04", "2026-03-15"]
    # Cột ngày kiểu Việt Nam (ngày trước) giữ nguyên cách đọc
    df_vn = normalize_dataframe_nfc(pd.DataFrame({"Ngày": ["03/04/2026", "25/04/2026"]}))
    assert df_vn["Ngày"].tolist() == ["2026-04-03", "2026-04-25"]


def test_base_unit_conversion_from_readme_multi_level():
    readme = "- **Quy đổi đơn vị**:\n- Bia Tiger lon 330ml: 1 thùng = 4 lốc, 1 lốc = 6 lon\n"
    sales = pd.DataFrame({
        "Mã đơn": ["A", "B", "C"],
        "Ngày": ["2026-01-01"] * 3,
        "Tên hàng": ["Bia Tiger lon 330ml"] * 3,
        "ĐVT": ["thùng", "lốc", "lon"],
        "Số lượng": [2, 1, 5],
    })
    out = BusinessDataStore._apply_base_unit_conversion({"ban": sales}, readme)["ban"]
    assert out["qty_base"].tolist() == [48.0, 6.0, 5.0]
    assert out["Đơn vị cơ bản"].tolist() == ["lon", "lon", "lon"]


def test_base_unit_conversion_from_conversion_table_by_code():
    conv = pd.DataFrame({
        "Mã hàng": ["SP1"], "Tên hàng": ["Sữa 180ml"], "Đơn vị lớn": ["Thùng"],
        "Đơn vị cơ bản": ["Hộp"], "Hệ số quy đổi": [48],
    })
    sales = pd.DataFrame({
        "Mã đơn": ["X", "Y"], "Ngày": ["2026-01-01"] * 2, "Mã hàng": ["SP1", "SP1"],
        "Tên hàng": ["Sữa 180ml"] * 2, "Đơn vị": ["thùng", "hộp"], "Số lượng": [1, 3],
    })
    out = BusinessDataStore._apply_base_unit_conversion({"quy_doi": conv, "ban": sales}, "")["ban"]
    assert out["qty_base"].tolist() == [48.0, 3.0]


def test_relevant_rules_keep_topic_and_core_sections():
    re_ = RuleEngine(ASSETS / "rulebook.md")
    rules = re_.select_relevant_rules("Lãi gộp tháng 5 là bao nhiêu?", "")
    assert "## 5. Kho, nhập hàng, giá vốn" in rules and "## 1. Đơn hàng và doanh thu" in rules
    rules_ph = re_.select_relevant_rules("Liệt kê các thuốc sắp hết hạn trong 30 ngày tới", "Nhà thuốc")
    assert "### Nhà thuốc" in rules_ph and "## 2. Thời gian" in rules_ph
    assert "(trích)" in rules_ph


def test_rule_selection_does_not_depend_on_section_numbers(tmp_path):
    rb = tmp_path / "rb.md"
    rb.write_text(
        "# Rulebook khác\nHôm nay là as_of_date.\n\n## A. Công nợ khách\nNợ tính theo đơn.\n\n"
        "## B. Mốc thời gian\nTuần bắt đầu **Chủ nhật**.\n\n## C. Theo ngành\n### Tiệm sách\n| Sách cũ | giảm 50% |\n",
        encoding="utf-8",
    )
    re_ = RuleEngine(rb)
    rules = re_.select_relevant_rules("Khách nào còn nợ nhiều nhất?", "Tiệm sách cũ")
    assert "## A. Công nợ khách" in rules and "## B. Mốc thời gian" in rules and "### Tiệm sách" in rules
    assert "Quy tắc Thời gian & Ranh giới" not in rules
    assert re_.knobs.week_start_day == "Chủ nhật"


class _FakeStore:
    def __init__(self, cols):
        self.cols = cols

    def get_shop_date_range(self, bid):
        return ("2025-01-01", "2026-12-31")

    def get_readme(self, bid):
        return ""

    def load_tables(self, bid):
        return {"ban_hang": pd.DataFrame(columns=self.cols)}


def test_unanswerable_keywords_respect_names_and_columns():
    dr = DateResolver(ASSETS / "calendar.csv")
    q = {"business_id": "b", "as_of_date": "2026-06-10"}
    is_u, _ = RuleEngine.detect_unanswerable_deterministic(
        dict(q, question="Khách Nam mua bao nhiêu tiền tháng này?"), _FakeStore(["Khách hàng", "Ngày"]), dr)
    assert not is_u
    is_u, _ = RuleEngine.detect_unanswerable_deterministic(
        dict(q, question="Nhân viên nào được đánh giá sao cao nhất?"), _FakeStore(["Nhân viên", "Đánh giá sao", "Ngày"]), dr)
    assert not is_u
    is_u, _ = RuleEngine.detect_unanswerable_deterministic(
        dict(q, question="Nhân viên nào thái độ tốt nhất?"), _FakeStore(["Nhân viên", "Ngày"]), dr)
    assert is_u


def test_scope_clause_question_expects_single_string():
    assert RuleEngine.infer_expected_type("Trong các món, món nào bán chạy nhất tháng này?") == "string"
    assert RuleEngine.infer_expected_type("Những khách nào mua trên 5 đơn tháng này?") == "list"


def test_string_with_comma_not_split_and_month_answer_format():
    p = postprocess_and_guard("q", "Trong các món, món nào bán chạy nhất?", {"type": "string", "answer": "Bánh mì thịt, chả"}, "string")
    assert p == {"id": "q", "type": "string", "answer": "Bánh mì thịt, chả"}
    p = postprocess_and_guard("q", "Tháng nào năm 2025 có doanh thu cao nhất?", {"type": "date", "answer": "2025-07-01"}, "date")
    assert p["answer"] == "2025-07"


def test_return_status_not_propagated_to_unrelated_table_with_same_ids():
    sales = normalize_dataframe_nfc(pd.DataFrame({
        "Mã đơn": ["1", "2", "3"], "Ngày": ["2026-01-01"] * 3,
        "Trạng thái": ["Trả hàng", "Hoàn thành", "Hoàn thành"], "Thành tiền": [1, 2, 3],
    }))
    imports = normalize_dataframe_nfc(pd.DataFrame({
        "Số phiếu": ["1", "10", "11", "12"], "Ngày": ["2026-01-01"] * 4,
        "Trạng thái": ["Hoàn thành"] * 4, "Tổng tiền": [5, 6, 7, 8],
    }))
    out = BusinessDataStore._standardize_business_tables({"ban": sales, "nhap": imports})
    assert out["nhap"]["Trạng thái"].tolist() == ["Hoàn thành"] * 4
    assert out["ban"]["Trạng thái"].tolist()[0] == "Trả hàng"


# ---------------------------------------------------------------------------
# Đợt rà soát 3
# ---------------------------------------------------------------------------

def _periods(q, as_of="2026-06-10"):
    dr = DateResolver(ASSETS / "calendar.csv")
    return [(p["start_date"], p["end_date"]) for p in dr.resolve(q, as_of)["detected_periods"]]


def test_month_with_relative_year():
    assert _periods("Doanh thu tháng 3 năm ngoái là bao nhiêu?") == [("2025-03-01", "2025-03-31")]
    assert _periods("Quý 2 năm ngoái doanh thu?") == [("2025-04-01", "2025-06-30")]


def test_full_date_range_has_no_spurious_month():
    assert _periods("Doanh thu từ ngày 1/5/2026 đến 10/5/2026?") == [("2026-05-01", "2026-05-10")]


def test_more_relative_periods():
    assert _periods("Tuần trước nữa bán bao nhiêu?") == [("2026-05-25", "2026-05-31")]
    assert _periods("Doanh thu quý này?") == [("2026-04-01", "2026-06-10")]
    assert _periods("Doanh thu quý trước?") == [("2026-01-01", "2026-03-31")]
    assert _periods("Từ đầu năm đến nay doanh thu bao nhiêu?") == [("2026-01-01", "2026-06-10")]
    assert _periods("Doanh thu năm 2025?") == [("2025-01-01", "2025-12-31")]
    assert _periods("Doanh thu ngày 5/6 là bao nhiêu?") == [("2026-06-05", "2026-06-05")]
    assert _periods("Doanh thu từ 01/05 đến 15/05?") == [("2026-05-01", "2026-05-15")]


def test_names_differing_in_diacritics_are_not_merged():
    t = BusinessDataStore._standardize_business_tables({"ban": pd.DataFrame({
        "Tên hàng": ["Bánh mì bơ", "Bánh mì bò", "banh mi bo", "Trà sữa", "tra sua", "TRÀ  SỮA"],
        "Số lượng": [1] * 6,
    })})
    assert t["ban"]["Tên hàng"].tolist() == ["Bánh mì bơ", "Bánh mì bò", "banh mi bo", "Trà sữa", "Trà sữa", "Trà sữa"]
    c = BusinessDataStore._standardize_business_tables({"ban": pd.DataFrame({
        "Khách hàng": ["Anh Bảo", "Anh Bào", "anh bảo", "KL"], "Số lượng": [1] * 4,
    })})
    assert c["ban"]["Khách hàng"].tolist() == ["Anh Bảo", "Anh Bào", "Anh Bảo", "Khách lẻ"]


def test_vietnamese_number_formats(tmp_path):
    from src.ingest import _read_csv_robust, parse_vn_numeric_value
    f = tmp_path / "n.csv"
    f.write_text('Mã đơn,Đơn giá,Khối lượng,Số lượng (kg)\nA,55.000,1.5,"1,5"\nB,1.200.000,2.25,"2,25"\n', encoding="utf-8")
    df = normalize_dataframe_nfc(_read_csv_robust(f))
    assert df["Đơn giá"].tolist() == [55000.0, 1200000.0]
    assert df["Khối lượng"].tolist() == [1.5, 2.25]
    assert df["Số lượng (kg)"].tolist() == [1.5, 2.25]
    assert parse_vn_numeric_value("12,75") == 12.75
    assert parse_vn_numeric_value("1,500") == 1500.0


def test_money_average_rounds_to_dong():
    p = postprocess_and_guard("q", "Trung bình mỗi đơn khách chi bao nhiêu?", {"type": "number", "answer": 125432.6}, "number")
    assert p["answer"] == 125433


# ---------------------------------------------------------------------------
# Đợt rà soát 4
# ---------------------------------------------------------------------------

def test_cache_not_shared_between_splits_with_same_business_id(tmp_path):
    for split, val in (("train", 111), ("test", 999)):
        d = tmp_path / split / "data" / "biz_001"
        d.mkdir(parents=True)
        pd.DataFrame({"Mã đơn": ["X"], "Ngày": ["2026-05-01"], "Thành tiền": [val]}).to_csv(d / f"ban_{split}.csv", index=False)
    cache = tmp_path / "shared_cache"
    t_train = BusinessDataStore(tmp_path / "train" / "data", cache_root=cache).load_tables("biz_001")
    t_test = BusinessDataStore(tmp_path / "test" / "data", cache_root=cache).load_tables("biz_001")
    assert list(t_train) == ["ban_train"] and list(t_test) == ["ban_test"]
    # Cùng nguồn -> tái dùng cache
    assert list(BusinessDataStore(tmp_path / "train" / "data", cache_root=cache).load_tables("biz_001")) == ["ban_train"]


def test_excel_month_sheets_merged_even_when_catalog_sheet_first(tmp_path):
    d = tmp_path / "data" / "biz_x"
    d.mkdir(parents=True)
    with pd.ExcelWriter(d / "SoBanHang.xlsx") as w:
        pd.DataFrame({"Mã hàng": ["SP1"], "Tên hàng": ["Bia"], "Giá bán": [10000]}).to_excel(w, sheet_name="Danh mục", index=False)
        for m in (1, 2, 3):
            pd.DataFrame({"Mã đơn": [f"HD{m}"], "Ngày": [f"2026-0{m}-05"], "Mã hàng": ["SP1"], "Số lượng": [1], "Thành tiền": [10000]}).to_excel(
                w, sheet_name=f"Tháng {m}", index=False)
    tables = BusinessDataStore(tmp_path / "data", cache_root=tmp_path / "c").load_tables("biz_x")
    assert set(tables) == {"Danh mục", "SoBanHang"}
    assert len(tables["SoBanHang"]) == 3


def test_sandbox_serializes_duckdb_decimal_as_number():
    import json
    from decimal import Decimal
    from src.sandbox import _serialize_default
    out = json.loads(json.dumps([Decimal("1234.50"), Decimal("300.0")], default=_serialize_default))
    assert out == [1234.5, 300]


def test_type_coercion_to_expected_type():
    assert postprocess_and_guard("q", "Tỉ lệ đơn hủy là bao nhiêu?", {"type": "number", "answer": 12.5}, "percent")["type"] == "percent"
    assert postprocess_and_guard("q", "Ngày nào bán nhiều nhất?", {"type": "string", "answer": "2026-05-12"}, "date")["type"] == "date"
    assert postprocess_and_guard("q", "Có cao hơn không?", {"type": "string", "answer": "có"}, "boolean")["answer"] is True
    assert postprocess_and_guard("q", "Món nào bán chạy nhất?", {"type": "list", "answer": ["Trà sữa"]}, "string") == {
        "id": "q", "type": "string", "answer": "Trà sữa"}
    assert postprocess_and_guard("q", "Những món nào bán trên 100 ly?", {"type": "list", "answer": ["Trà sữa"]}, "list")["type"] == "list"


# ---------------------------------------------------------------------------
# Gỡ gắn cứng theo rulebook train
# ---------------------------------------------------------------------------

def test_prompts_contain_no_train_rulebook_rules_and_forbid_comments():
    from model import baseline
    for prompt in (baseline.SYSTEM_PROMPT_SQL, baseline.SYSTEM_PROMPT_DEFENSIVE_SQL, baseline.SYSTEM_PROMPT_PLAN_SQL):
        for train_rule in ("ĐANG GIAO VẪN TÍNH", "Phí phục vụ", "COD", "Khách lẻ", "'Hoàn thành'\n", "docstring ngắn"):
            assert train_rule not in prompt
        assert "KHÔNG viết bất kỳ chú thích nào" in prompt


def test_fewshot_examples_have_no_comments_or_business_rules():
    from src.rule_engine import DAILFewShotSelector as D
    for name in D.EXEMPLARS:
        text = D.EXEMPLARS[name].format(valid=D.VALID)
        assert "--" not in text and "/*" not in text and "#" not in text
        assert "Hoàn thành" not in text and "Đã hủy" not in text


def test_date_resolver_uses_rulebook_week_start_and_tet(tmp_path):
    from types import SimpleNamespace
    dr = DateResolver(ASSETS / "calendar.csv")
    dr.apply_knobs(SimpleNamespace(week_start_day="Chủ nhật", tet_start_lunar_month=12, tet_start_lunar_day=25,
                                   tet_end_lunar_month=1, tet_end_lunar_day=7))
    assert dr.resolve("tuần này", "2026-06-10")["this_week"] == ("2026-06-07", "2026-06-10")
    default = DateResolver(ASSETS / "calendar.csv")
    assert dr.get_tet_range(2026)[0] > default.get_tet_range(2026)[0]
    assert dr.get_tet_range(2026)[1] < default.get_tet_range(2026)[1]


def test_walk_in_rule_follows_rulebook(tmp_path):
    rb = tmp_path / "rb.md"
    rb.write_text("# R\n## 1. Khách\nMọi đơn đều có khách, kể cả khách lẻ.\n", encoding="utf-8")
    assert RuleEngine(rb).walk_in_not_customer is False
    assert RuleEngine(ASSETS / "rulebook.md").walk_in_not_customer is True
    p = postprocess_and_guard("q", "Khách nào mua nhiều nhất?", {"type": "string", "answer": "Khách lẻ"}, "string", reject_walk_in=False)
    assert p["answer"] == "Khách lẻ"


def test_catalog_and_returns_detected_by_structure_not_table_name():
    sales = normalize_dataframe_nfc(pd.DataFrame({
        "Mã đơn": ["D1", "D2"], "Ngày": ["2026-01-01"] * 2, "Tên hàng": ["tra sua", "Cà phê"],
        "Trạng thái": ["Hoàn thành", "Hoàn thành"], "Thành tiền": [1, 2],
    }))
    catalog = pd.DataFrame({"Mã hàng": ["A", "B"], "Tên hàng": ["Trà Sữa", "Cà phê"]})
    returns = pd.DataFrame({"Mã phiếu trả": ["R1"], "Mã đơn gốc": ["D2"], "Ngày": ["2026-01-02"]})
    out = BusinessDataStore._standardize_business_tables({"ds_mon_ban": catalog, "phieu_tra_hang_2026": returns, "ban_le": sales})
    assert out["ban_le"]["Tên hàng"].tolist() == ["Trà Sữa", "Cà phê"]
    assert out["ban_le"]["Trạng thái"].tolist() == ["Hoàn thành", "Trả hàng"]


def test_period_split_tables_merged_generically():
    a = pd.DataFrame({"Mã đơn": ["1"], "Ngày": ["2026-01-01"], "Thành tiền": [1]})
    b = pd.DataFrame({"Mã đơn": ["2"], "Ngày": ["2026-02-01"], "Thành tiền": [2]})
    out = BusinessDataStore._merge_period_split_tables({"SoQuy_T1": a, "SoQuy_T2": b, "Khac": a})
    assert set(out) == {"SoQuy", "Khac"} and len(out["SoQuy"]) == 2


# ---------------------------------------------------------------------------
# Giải mã Telex theo luật (không dùng danh sách từ lỗi)
# ---------------------------------------------------------------------------

def test_telex_decoder_rules():
    from src.telex import decode_telex_text, decode_telex_word
    expected = {"naof": "nào", "thangs": "tháng", "cuwar": "cửa", "tieenf": "tiền", "nhieefu": "nhiều", "ddieenj": "điện",
                "gias": "giá", "tawng": "tăng", "giamr": "giảm", "vaayj": "vậy", "heets": "hết", "nhaats": "nhất"}
    for raw, dec in expected.items():
        assert decode_telex_word(raw) == dec
    # từ tiếng Việt không dấu / tiếng Anh / mã sản phẩm không bị đổi
    for w in ("bao", "ngay", "tong", "dthu", "tiger", "green", "coffee", "kg"):
        assert decode_telex_word(w) is None
    assert decode_telex_text("Bia Tiger Max bán thangs naof", {"max"}) == "Bia Tiger Max bán tháng nào"
    assert decode_telex_text('ghi "thangs naof" nguyên', None) == 'ghi "thangs naof" nguyên'
    assert decode_telex_text("naof thangs", {"thangs"}) == "nào thangs"


class _StoreWithCategories(_FakeStore):
    def __init__(self, cols, cats=(), rng=("2025-01-01", "2026-12-31")):
        super().__init__(cols)
        self.cats, self.rng = set(cats), rng

    def get_shop_date_range(self, bid):
        return self.rng

    def get_category_values(self, bid):
        return self.cats


def _unans(q, store, as_of="2026-06-10"):
    dr = DateResolver(ASSETS / "calendar.csv")
    return RuleEngine.detect_unanswerable_deterministic({"question": q, "business_id": "b", "as_of_date": as_of}, store, dr)[0]


def test_medical_keyword_that_is_a_product_group_is_not_unanswerable():
    cols = ["Nhóm hàng", "Thành tiền", "Ngày"]
    assert not _unans("Nhóm dị ứng chiếm bao nhiêu phần trăm doanh thu tháng 5/2026?", _StoreWithCategories(cols, cats={"di ung", "thuoc"}))
    assert _unans("Thuốc nào có tác dụng phụ gây dị ứng?", _StoreWithCategories(cols, cats={"thuoc"}))


def test_comparison_baseline_outside_data_is_unanswerable():
    store = _StoreWithCategories(["Thành tiền", "Ngày"], rng=("2025-02-01", "2026-06-30"))
    assert _unans("Tháng 2/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?", store)
    assert not _unans("Tháng 3/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?", store)
    assert _unans("Tháng 5/2025 doanh thu tăng bao nhiêu % so với cùng kỳ năm ngoái?", store)


def test_expiry_question_with_future_date_is_not_out_of_range():
    store = _StoreWithCategories(["Hạn sử dụng", "Tồn kho", "Ngày"], rng=("2025-01-01", "2026-05-25"))
    assert not _unans("Có mấy lô hàng sẽ hết hạn từ nay đến hết ngày 22/09/2026?", store, as_of="2026-05-25")


def test_lowercase_gender_vs_capitalized_name():
    store = _StoreWithCategories(["Khách hàng", "Thành tiền", "Ngày"])
    assert _unans("tháng 11/2025 khách nam mua tổng cộng bao nhiêu tiền?", store)
    assert not _unans("Tháng 11/2025 khách Nam mua tổng cộng bao nhiêu tiền?", store)


def test_threshold_columns_are_not_stock_data():
    cols = ["Tên hàng", "Tồn nhỏ nhất", "Tồn lớn nhất", "Ngày", "Thành tiền"]
    assert _unans("Hết hôm nay trong kho còn bao nhiêu hộp bánh?", _StoreWithCategories(cols))


def test_topic_bank_requires_missing_column_and_respects_word_boundaries():
    base = ["Khách hàng", "Thành tiền", "Ngày"]
    assert not _unans("Doanh thu tháng 5/2026 là bao nhiêu?", _StoreWithCategories(base))
    cases = [
        ("Khách hàng nào được khen nhiều nhất tháng 5/2026?", ["đánh giá"]),
        ("Mỗi khách bao nhiêu tuổi vậy?", ["ngày sinh"]),
        ("Giá thị trường của bia Tiger hiện nay là bao nhiêu?", ["giá tham khảo"]),
        ("Dự đoán doanh thu tháng sau?", ["dự báo"]),
        ("Tiền lương nhân viên tháng 5/2026 là bao nhiêu?", ["lương"]),
        ("Mặt hàng X có xuất xứ ở đâu?", ["xuất xứ"]),
    ]
    for q, present_col in cases:
        assert _unans(q, _StoreWithCategories(base)), q
        assert not _unans(q, _StoreWithCategories(base + present_col)), q


def test_topic_keyword_defined_in_rulebook_is_never_flagged():
    dr = DateResolver(ASSETS / "calendar.csv")
    store = _StoreWithCategories(["Khách hàng", "Thành tiền", "Ngày"])
    q = {"question": "Ai là nhân viên chăm chỉ nhất? Đi làm muộn mấy ngày?", "business_id": "b", "as_of_date": "2026-06-10"}
    assert RuleEngine.detect_unanswerable_deterministic(q, store, dr)[0]
    rb = "## Xếp hạng\n| Chăm chỉ nhất | nhiều ngày làm việc nhất |\n| đi làm muộn | tính theo giờ tạo đơn |"
    assert not RuleEngine.detect_unanswerable_deterministic(q, store, dr, rulebook_text=rb)[0]


def test_unaccented_multiword_keyword_matches():
    assert _unans("khach hang nao co thai do tot nhat thang 5/2026", _StoreWithCategories(["Khách hàng", "Thành tiền", "Ngày"]))


# ---------------------------------------------------------------------------
# Từ điển (assets/lexicon.json)
# ---------------------------------------------------------------------------

def test_lexicon_query_expansion_contextual():
    from src.lexicon import load_lexicon
    L = load_lexicon()
    assert L.expand_query("so vs th trc dthu hnay bn") == "so với tháng trước doanh thu hôm nay bao nhiêu"
    assert L.expand_query("đơn trả bằng tm và bằng ck") == "đơn trả bằng tiền mặt và bằng chuyển khoản"
    assert L.expand_query("mức ck 5% hôm qua") == "mức chiết khấu 5% hôm qua"
    # mã có số, chuỗi trong ngoặc kép, từ lạ không bị đổi
    assert L.expand_query("đơn dc 2 và \"dc\" ko dc") == "đơn dc 2 và \"dc\" không được"
    assert L.expand_query("Bia Tiger Max th5") == "Bia Tiger Max th5"


def test_lexicon_status_variants_and_value_hints():
    from src.lexicon import load_lexicon
    L = load_lexicon()
    assert normalize_order_status("Huỷ đơn") == "Đã hủy"
    assert normalize_order_status("hàng trả về") == "Trả hàng"
    assert normalize_order_status("Đang kiểm tra") == "Đang kiểm tra"
    tables = {"don_hang": pd.DataFrame({"Trả bằng": ["cash", "cod", "debt", "cash"], "Kênh": ["store", "phone", "store", "zalo_oa"]})}
    hint = L.build_value_hints("doanh thu các đơn trả bằng tiền mặt bán tại quầy", tables)
    assert "'cash'" in hint and "'store'" in hint
    assert L.build_value_hints("doanh thu hôm nay", tables) == ""
    # shop không có cột mã hoá tương ứng -> không gợi ý
    assert L.build_value_hints("đơn trả bằng tiền mặt", {"x": pd.DataFrame({"A": ["foo", "bar"]})}) == ""


def test_lexicon_topics_loaded_and_train_only_guard(tmp_path):
    import subprocess
    import sys
    from src.lexicon import load_lexicon
    assert len(load_lexicon().unanswerable_topics) >= 10
    fake_public = tmp_path / "public_test"
    (fake_public / "data").mkdir(parents=True)
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(root / "tools" / "mine_lexicon.py"), "--train-root", str(fake_public)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "TRAIN" in (r.stdout + r.stderr)


def test_entity_index_survives_name_column_that_is_also_unit_column():
    from src.rule_engine import DynamicEntityGrounder
    df = pd.DataFrame({"Hàng - Unit": ["Bia Tiger thùng", "Bia Tiger lon", "Nước suối chai"], "Số lượng": [1, 2, 3]})
    hint = DynamicEntityGrounder().find_grounding_hints("bán bao nhiêu bia tiger lon", "b", {"ban": df})
    assert "Bia Tiger lon" in hint
    with_unit = pd.DataFrame({"Tên hàng": ["Bia Tiger", "Nước suối"], "Đơn vị tính": ["lon", "chai"]})
    hint2 = DynamicEntityGrounder().find_grounding_hints("doanh thu nuoc suoi", "b2", {"ban": with_unit})
    assert "Nước suối" in hint2 and "chai" in hint2


# ---------------------------------------------------------------------------
# Đọc dữ liệu preprocessed (<tập>/<shop>/tables/*.csv + _readme.txt)
# ---------------------------------------------------------------------------

def _write_preprocessed_shop(root, shop="s1", readme="# s\n\n## File\n- `a.xlsx` (xlsx)\n", dup_rows=0):
    d = root / shop
    (d / "tables").mkdir(parents=True)
    (d / "_readme.txt").write_text(readme, encoding="utf-8")
    parts = []
    for m, year in ((1, 2025), (2, 2025), (3, 2026), (4, 2026)):
        n = 4
        df = pd.DataFrame({
            "Mã đơn hàng": [f"{year}{m}{i}" for i in range(n)],
            "Ngày": [f"{year}-0{m}-0{i + 1}" for i in range(n)],
            "Tên hàng": ["Trà sữa", "tra sua", "Cà phê", "Cà phê"],
            "Số lượng": [1, 2, 3, 4],
            "Thành tiền": [10, 20, 30, 40],
            "Trạng thái": ["Hoàn thành"] * n,
        })
        parts.append(df)
        df.to_csv(d / "tables" / f"BanHang_T{m}_{year}.csv", index=False)
    union = pd.concat(parts, ignore_index=True)
    union["Tên hàng"] = union["Tên hàng"].replace({"tra sua": "Trà sữa"})  # bảng gộp đã chuẩn hoá cách viết tên
    if dup_rows:
        union = pd.concat([union, union.head(dup_rows)], ignore_index=True)
    union.to_csv(d / "tables" / "BanHang.csv", index=False)
    pd.DataFrame({"Mã hàng": ["A", "B"], "Tên hàng": ["Trà sữa", "Cà phê"], "Giá": [1, 2]}).to_csv(d / "tables" / "DanhMuc.csv", index=False)
    return d


def test_preprocessed_layout_drops_union_duplicates_and_keeps_originals(tmp_path):
    root = tmp_path / "split_pre"
    _write_preprocessed_shop(root)
    tables = BusinessDataStore(root, cache_root=tmp_path / "c").load_tables("s1")
    assert set(tables) == {"BanHang", "DanhMuc"}
    assert len(tables["BanHang"]) == 16  # 4 bảng gốc x 4 dòng, KHÔNG phải 32 (đã bỏ bảng gộp trùng)


def test_preprocessed_union_with_extra_duplicates_is_not_kept(tmp_path):
    root = tmp_path / "split_pre"
    _write_preprocessed_shop(root, dup_rows=3)
    tables = BusinessDataStore(root, cache_root=tmp_path / "c").load_tables("s1")
    assert len(tables["BanHang"]) == 16


def test_preprocessed_catalog_table_not_mistaken_for_part(tmp_path):
    root = tmp_path / "split_pre"
    _write_preprocessed_shop(root)
    tables = BusinessDataStore(root, cache_root=tmp_path / "c").load_tables("s1")
    assert "DanhMuc" in tables and len(tables["DanhMuc"]) == 2


def test_readme_sqlite_only_detection():
    f = BusinessDataStore._readme_sources_sqlite_only
    assert f("## File\n- `pos.sqlite` (sqlite)\n")
    assert not f("## File\n- `a.csv` (csv)\n- `b.sqlite` (sqlite)\n")
    assert not f("không có danh sách file")


def test_default_cache_root_by_layout(tmp_path):
    raw_layout = BusinessDataStore(tmp_path / "train" / "data")
    pre_layout = BusinessDataStore(tmp_path / "public_test_preprocessed")
    assert raw_layout.cache_root == tmp_path / ".cache" / "ingested"
    assert pre_layout.cache_root == tmp_path / ".cache" / "ingested"
