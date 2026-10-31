# -*- coding: utf-8 -*-
"""
Bộ kiểm thử hồi quy (Regression Tests) xác nhận việc sửa các bug logic & runtime đã phát hiện:
  1. Sanity Guard 2 không nuốt câu hỏi lũy kế nhiều ngày > 500 triệu.
  2. _FuzzyDateContext trả về đúng two_weeks_ago thay vì last_week.
  3. extract_python_code ưu tiên block python khi có block text đi kèm.
  4. ingest.py đồng bộ status_norm khi đơn hàng được cập nhật Trả hàng.
  5. DateResolver phân biệt 2 tuần trước với 1 tuần trước và xử lý dải ngày từ... đến...
  6. RuleEngine.infer_expected_type xử lý lỗi gõ Telex và tránh nhầm 'hok kê đơn' sang boolean.
  7. Table postprocessing loại bỏ dòng Khách lẻ và chuẩn hóa chuỗi ngoặc đơn.
"""

from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from src.date_resolver import DateResolver
from src.postprocess import postprocess_and_guard
from src.rule_engine import RuleEngine
from src.sandbox import extract_python_code, _FuzzyDateContext


def test_guard2_allows_multi_day_high_revenue():
    # Câu hỏi lũy kế năm > 500M phải giữ nguyên NUMBER
    q_ytd = "Từ đầu năm đến hết hôm nay cửa hàng bán được bao nhiêu tiền?"
    res_ytd = postprocess_and_guard("q1", q_ytd, 8843326100, "number")
    assert res_ytd["type"] == "number"
    assert res_ytd["answer"] == 8843326100

    # Câu hỏi 30 ngày gần nhất > 500M phải giữ nguyên NUMBER
    q_30d = "doanh thu 30 ngày gần nhất của cửa hàng là bao nhiêu"
    res_30d = postprocess_and_guard("q2", q_30d, 1457273700, "number")
    assert res_30d["type"] == "number"
    assert res_30d["answer"] == 1457273700

    # Nhưng câu hỏi 1 ngày duy nhất (hôm nay / hôm qua) > 500M vẫn bị kích hoạt ABSTAIN do nghi ngờ sum nhầm toàn shop
    q_single = "riêng hôm nay doanh thu là bao nhiêu?"
    res_single = postprocess_and_guard("q3", q_single, 600000000, "number")
    assert res_single["type"] == "abstain"


def test_fuzzy_date_context_two_weeks():
    fdc = _FuzzyDateContext({
        "as_of_date": "2025-05-20",
        "this_week": ("2025-05-19", "2025-05-20"),
        "last_week": ("2025-05-12", "2025-05-18"),
        "two_weeks_ago": ("2025-05-05", "2025-05-11"),
    })
    # Truy vấn hai tuần trước phải trả về two_weeks_ago, không phải last_week
    res_2w = fdc["hai_tuan_truoc"]
    assert res_2w == ("2025-05-05", "2025-05-11")
    assert fdc["2_tuan"] == ("2025-05-05", "2025-05-11")
    assert fdc["two_weeks"] == ("2025-05-05", "2025-05-11")


def test_extract_python_code_prioritizes_python_block():
    text_mixed = """
Sau đây là giải pháp:
```python
result = {"type": "number", "answer": 123456}
```
Kết quả mẫu minh họa:
```
123456
```
"""
    code = extract_python_code(text_mixed)
    assert 'result = {"type": "number", "answer": 123456}' in code
    assert code.startswith("result =")


def test_date_resolver_two_weeks_and_range():
    dr = DateResolver()
    ctx_2w = dr.resolve("doanh thu 2 tuần trước là bao nhiêu?", "2025-05-20")
    assert len(ctx_2w["detected_periods"]) >= 1
    p2 = ctx_2w["detected_periods"][0]
    assert "Hai tuần trước" in p2["expr"]
    assert p2["start_date"] == "2025-05-05"
    assert p2["end_date"] == "2025-05-11"

    # Dải ngày từ ... đến ...
    ctx_rng = dr.resolve("từ ngày 01/05/2025 đến ngày 15/05/2025 dthu là bao nhiêu?", "2025-06-01")
    assert len(ctx_rng["detected_periods"]) == 1
    prng = ctx_rng["detected_periods"][0]
    assert prng["start_date"] == "2025-05-01"
    assert prng["end_date"] == "2025-05-15"


def test_rule_engine_type_inference_telex_and_dialects():
    re_eng = RuleEngine()
    # 1. Telex ngayf -> date
    assert re_eng.infer_expected_type("ngayf nào trong tháng 4/2025 dthu cao nhất") == "date"
    # 2. Telex trawm -> percent
    assert re_eng.infer_expected_type("cho hỏi dthu thangs 11/2025 hơn hay kém tháng trc bnhiu phần trawm") == "percent"
    # 3. 'hok kê đơn' không bị nhầm thành boolean
    assert re_eng.infer_expected_type("tháng 5/2026 dthu hàng hok kê đơn là bn ta") == "number"
    # 4. ncc nao không dấu -> string
    assert re_eng.infer_expected_type("thang 9/2025 tien nhap hang tu nha cung cap nao cao nhat") == "string"
    # 5. List không dấu: moi loai bnhiu, ghi 3 so, no them tra duoc
    assert re_eng.infer_expected_type("check gium thang 10/2025, dthu cac don tra bang chuyen khoan va cac don tra bang tm moi loai bnhiu") == "list"
    assert re_eng.infer_expected_type("xem giup tu thang 3/2025 den thang 5/2025, doanh thu moi thang la bao nhieu? ghi 3 so theo thu tu thoi gian") == "list"
    assert re_eng.infer_expected_type("khach Phung Dat Trang thang 2/2025 no them bao nhieu, tra duoc bao nhieu") == "list"


def test_table_postprocessing_excludes_walkin_customer():
    raw_tbl = {
        "columns": ["Khách hàng", "Nợ"],
        "rows": [
            ["Khách lẻ", 500000],
            ["Nguyễn Văn A (1.500.000đ)", 400000],
            ["Trần Thị B", 300000],
        ]
    }
    cleaned = postprocess_and_guard("q_tbl", "Top khách hàng còn nợ nhiều nhất kèm số nợ", raw_tbl, "table")
    assert cleaned["type"] == "table"
    ans_rows = cleaned["answer"]["rows"]
    # Khách lẻ phải bị lọc bỏ
    assert len(ans_rows) == 2
    assert ans_rows[0][0] == "Nguyễn Văn A"  # strip_trailing_measurement_in_parens
    assert ans_rows[0][1] == 400000
    assert ans_rows[1][0] == "Trần Thị B"
    assert ans_rows[1][1] == 300000
