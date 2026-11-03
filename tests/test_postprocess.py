# -*- coding: utf-8 -*-
"""
Unit tests cho Module 5: Post-Processing & Penalized Accuracy Guard.
Kiểm tra làm tròn chuẩn Mục 9 (nửa đồng làm tròn lên), định dạng kiểu đáp án, và check_consensus.
"""

import pytest
from src.postprocess import (
    round_half_up_int,
    round_half_up_2dec,
    postprocess_and_guard,
    check_consensus,
    are_strings_in_consensus,
)


def test_round_half_up_int():
    # Nửa đồng làm tròn lên theo Mục 3 & Mục 9 Rulebook
    assert round_half_up_int(100.4) == 100
    assert round_half_up_int(100.5) == 101
    assert round_half_up_int(100.6) == 101
    assert round_half_up_int(0.5) == 1
    assert round_half_up_int(0.0) == 0


def test_round_half_up_2dec():
    assert round_half_up_2dec(12.344) == 12.34
    assert round_half_up_2dec(12.345) == 12.35
    assert round_half_up_2dec(12.346) == 12.35
    assert round_half_up_2dec(10.0) == 10


def test_postprocess_and_guard_number_and_percent():
    # Tiền VND: làm tròn nguyên
    p_money = postprocess_and_guard("q1", "Doanh thu là bao nhiêu tiền?", 4500000.5, "number")
    assert p_money == {"id": "q1", "type": "number", "answer": 4500001}

    # Số lượng đếm hoặc trung bình:
    p_cnt = postprocess_and_guard("q2", "Bình quân mỗi đơn có bao nhiêu ly?", 4.567, "number")
    assert p_cnt == {"id": "q2", "type": "number", "answer": 4.57}

    # Percent: làm tròn 2 chữ số
    p_pct = postprocess_and_guard("q3", "Tỉ lệ hủy là bao nhiêu %?", 12.3456, "percent")
    assert p_pct == {"id": "q3", "type": "percent", "answer": 12.35}


def test_postprocess_and_guard_date_and_boolean():
    p_date = postprocess_and_guard("q4", "Ngày nào?", "2026-05-15", "date")
    assert p_date == {"id": "q4", "type": "date", "answer": "2026-05-15"}

    p_bool = postprocess_and_guard("q5", "Có tăng không?", True, "boolean")
    assert p_bool == {"id": "q5", "type": "boolean", "answer": True}


def test_postprocess_and_guard_list_and_table():
    p_list = postprocess_and_guard("q6", "Top sản phẩm", ["Trà sữa", "Cà phê"], "list")
    assert p_list == {"id": "q6", "type": "list", "answer": ["Trà sữa", "Cà phê"]}

    tbl_raw = {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 1500000]]}
    p_tbl = postprocess_and_guard("q7", "Bảng nợ", tbl_raw, "table")
    assert p_tbl == {"id": "q7", "type": "table", "answer": tbl_raw}


def test_postprocess_and_guard_abstain_on_invalid():
    # NaN hoặc None -> Safe Abstain
    assert postprocess_and_guard("q8", "Tiền?", None, "number") == {"id": "q8", "type": "abstain"}
    assert postprocess_and_guard("q9", "Tiền?", float("nan"), "number") == {"id": "q9", "type": "abstain"}
    # Doanh thu âm vô lý -> Safe Abstain
    assert postprocess_and_guard("q10", "Tổng doanh thu bao nhiêu?", -500, "number") == {"id": "q10", "type": "abstain"}


def test_check_consensus():
    # 1. Số khớp nhau
    p1 = {"type": "number", "answer": 1000000}
    p2 = {"type": "number", "answer": 1000000}
    assert check_consensus(p1, p2)

    # 2. Số khác nhau -> Bất đồng
    p3 = {"type": "number", "answer": 1000000}
    p4 = {"type": "number", "answer": 1500000}
    assert not check_consensus(p3, p4)

    # 3. Chuỗi tương đương (không phân biệt hoa thường, dấu, khoảng trắng)
    s1 = {"type": "string", "answer": "Trà Sữa Trân Châu"}
    s2 = {"type": "string", "answer": "trà sữa trân châu"}
    assert check_consensus(s1, s2)

    # 4. Table khớp nhau
    t1 = {"type": "table", "answer": {"columns": ["k", "v"], "rows": [["A", 100], ["B", 200]]}}
    t2 = {"type": "table", "answer": {"columns": ["k", "v"], "rows": [["B", 200], ["A", 100]]}}
    assert check_consensus(t1, t2)


def test_clean_entity_label_annotations():
    from src.postprocess import clean_entity_label
    # 1. Bỏ số lượng/doanh thu đính kèm trong ngoặc
    assert clean_entity_label("Trà đào cam sả (256)") == "Trà đào cam sả"
    assert clean_entity_label("Cà phê sữa đá (307 ly)") == "Cà phê sữa đá"
    assert clean_entity_label("Bánh tráng (1.605.000đ)") == "Bánh tráng"

    # 2. Giữ nguyên size F&B và quy cách dược phẩm
    assert clean_entity_label("Trà sữa trân châu (M)") == "Trà sữa trân châu (M)"
    assert clean_entity_label("Trà sữa trân châu (L)") == "Trà sữa trân châu (L)"
    assert clean_entity_label("Panadol Extra (500mg)") == "Panadol Extra (500mg)"
    assert clean_entity_label("Nước muối sinh lý (500ml)") == "Nước muối sinh lý (500ml)"


def test_postprocess_preserves_zero_and_wholesale_quantities():
    # Số 0 là đáp án hợp lệ (ví dụ: hôm nay đóng cửa không bán được đơn nào, doanh thu = 0)
    p_zero = postprocess_and_guard("q_zero", "Hôm nay doanh thu bán hàng là bao nhiêu tiền?", 0, "number")
    assert p_zero["type"] == "number"
    assert p_zero["answer"] == 0

    # 0.0% là đáp án hợp lệ (ví dụ: không có đơn hủy nào -> tỉ lệ hủy là 0.0%)
    p_zero_pct = postprocess_and_guard("q_zero_pct", "Tỉ lệ đơn hủy là bao nhiêu %?", 0.0, "percent")
    assert p_zero_pct["type"] == "percent"
    assert p_zero_pct["answer"] == 0.0

    # Bán sỉ số lượng lớn (> 20.000) không bị chặn oan uổng
    p_wholesale = postprocess_and_guard("q_ws", "Bán được bao nhiêu viên gạch?", 50000, "number")
    assert p_wholesale["type"] == "number"
    assert p_wholesale["answer"] == 50000


def test_postprocess_round_half_up_negative():
    # Nửa đơn vị âm làm tròn xa số 0: -1.5 -> -2, -2.5 -> -3
    assert round_half_up_int(-1.5) == -2
    assert round_half_up_int(-2.5) == -3
    assert round_half_up_int(-1.4) == -1


def test_are_strings_in_consensus_variations():
    # Kiểm tra đồng thuận giữa các chuỗi có sự khác biệt nhỏ về dấu, khoảng trắng, ngoặc kép
    assert are_strings_in_consensus('"Trà sữa"', 'Trà sữa')
    assert are_strings_in_consensus('Cà phê sữa đá  ', '  cà phê sữa đá')
    # Khác dấu là khác tên (luật chấm giữ dấu): không đồng thuận
    assert not are_strings_in_consensus('Đoàn Chuẩn', 'Doan Chuan')
    assert not are_strings_in_consensus('Bánh mì bơ', 'Bánh mì bò')
    assert are_strings_in_consensus('Cà phê-sữa', 'cà phê sữa')
    assert not are_strings_in_consensus('Trà đào', 'Trà vải')


def test_postprocess_table_dict_dataframe_cleaning():
    import pandas as pd
    df = pd.DataFrame({"Món": ["Bánh mì", "Xôi"], "Giá": [20000, 15000]})
    p = postprocess_and_guard("q_tbl", "Danh sách món và giá", df, "table")
    assert p["type"] == "table"
    assert p["answer"]["columns"] == ["Món", "Giá"]
    assert len(p["answer"]["rows"]) == 2


def test_postprocess_scalar_nested_tuple_unpacking():
    # DuckDB fetchall() trả về [(1500000,)]
    p_num = postprocess_and_guard("q_tup1", "Doanh thu bao nhiêu tiền?", [(1500000,)], "number")
    assert p_num["type"] == "number"
    assert p_num["answer"] == 1500000

    # Lồng nhau dạng list [[25.5]] cho percent
    p_pct = postprocess_and_guard("q_tup2", "Tỉ lệ hủy bao nhiêu %?", [[25.5]], "percent")
    assert p_pct["type"] == "percent"
    assert p_pct["answer"] == 25.5

    # Lồng nhau dạng [('2026-05-15',)] cho date
    p_date = postprocess_and_guard("q_tup3", "Ngày nào bán chạy nhất?", [("2026-05-15",)], "date")
    assert p_date["type"] == "date"
    assert p_date["answer"] == "2026-05-15"

    # Lồng nhau dạng [(True,)] cho boolean
    p_bool = postprocess_and_guard("q_tup4", "Có đúng không?", [(True,)], "boolean")
    assert p_bool["type"] == "boolean"
    assert p_bool["answer"] is True

    # Lồng nhau dạng [('Trà sữa',)] cho string
    p_str = postprocess_and_guard("q_tup5", "Món nào bán chạy nhất?", [("Trà sữa",)], "string")
    assert p_str["type"] == "string"
    assert p_str["answer"] == "Trà sữa"


def test_postprocess_string_currency_cleaning():
    p1 = postprocess_and_guard("q_curr1", "Tổng doanh thu bao nhiêu?", "1.500.000 VNĐ", "number")
    assert p1["type"] == "number"
    assert p1["answer"] == 1500000

    p2 = postprocess_and_guard("q_curr2", "Tổng doanh thu?", "2,450,000.00", "number")
    assert p2["type"] == "number"
    assert p2["answer"] == 2450000


def test_check_consensus_post_rounding():
    # Tiền lệch phần lẻ: sau postprocess (làm tròn về đồng, Mục 9) thì giống nhau
    q = "Doanh thu bao nhiêu tiền?"
    p1 = postprocess_and_guard("q", q, 5000000.4, "number")
    p2 = postprocess_and_guard("q", q, 5000000.1, "number")
    assert check_consensus(p1, p2)

    # 2 kết quả phần trăm làm tròn 2 chữ số thập phân giống nhau
    p3 = {"type": "percent", "answer": 12.341}
    p4 = {"type": "percent", "answer": 12.344}
    assert check_consensus(p3, p4)


def test_check_consensus_rejects_integer_rounding_for_percent_and_averages():
    # 12.4% và 12.0% chấm điểm khác nhau -> không được coi là đồng thuận
    assert not check_consensus({"type": "percent", "answer": 12.4}, {"type": "percent", "answer": 12.0})
    assert not check_consensus({"type": "number", "answer": 2.45}, {"type": "number", "answer": 2.1})


def test_numeric_list_keeps_duplicates():
    p = postprocess_and_guard("q", "Còn mấy thùng mấy chai?", {"type": "list", "answer": [3, 3]}, "list")
    assert p["answer"] == [3, 3]


def test_small_percent_not_rescaled_when_code_multiplies_by_100():
    raw = {"type": "percent", "answer": 0.45}
    code = "val = (a - b) * 100.0 / b"
    assert postprocess_and_guard("q", "Tăng bao nhiêu %?", raw, "percent", code=code)["answer"] == 0.45
    assert postprocess_and_guard("q", "Tăng bao nhiêu %?", raw, "percent", code="val = (a - b) / b")["answer"] == 45


def test_size_suffix_is_not_consensus():
    assert not are_strings_in_consensus("Trà sữa (M)", "Trà sữa (L)")
    assert are_strings_in_consensus("Trà sữa (M)", "trà sữa (m)")

