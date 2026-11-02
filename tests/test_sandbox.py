# -*- coding: utf-8 -*-
"""
Unit tests cho Module 4: Execution Sandbox & Safe Builtins.
Kiểm tra an ninh (ACE prevention), timeout an toàn và SafeDuckDBWrapper.
"""

import math
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.sandbox import (
    PythonSandbox,
    SafeDuckDBWrapper,
    make_safe_import,
    make_safe_builtins,
    extract_python_code,
    has_while_loop,
)


def test_safe_builtins_blocks_dangerous_functions():
    safe_imp = make_safe_import(duckdb, pd, np)
    b_dict = make_safe_builtins(safe_imp)

    # Chặn các hàm can thiệp file và thực thi mã động
    assert "open" not in b_dict
    assert "eval" not in b_dict
    assert "exec" not in b_dict
    assert "compile" not in b_dict
    assert "breakpoint" not in b_dict
    assert "input" not in b_dict

    # Giữ lại các hàm built-in toán học, cấu trúc dữ liệu an toàn
    assert "len" in b_dict
    assert "range" in b_dict
    assert "sum" in b_dict
    assert "min" in b_dict
    assert "max" in b_dict
    assert "sorted" in b_dict
    assert "dict" in b_dict
    assert "list" in b_dict
    assert "isinstance" in b_dict


def test_safe_import_whitelist():
    safe_imp = make_safe_import(duckdb, pd, np)

    # Cho phép các thư viện tính toán an toàn
    assert safe_imp("math").sqrt(16) == 4.0
    assert safe_imp("re").search(r"\d+", "abc123def").group(0) == "123"
    assert safe_imp("json").loads('{"a": 1}') == {"a": 1}

    # Chặn đứng các module nhạy cảm (Arbitrary Code Execution / Network)
    with pytest.raises(ImportError):
        safe_imp("os")

    with pytest.raises(ImportError):
        safe_imp("sys")

    with pytest.raises(ImportError):
        safe_imp("subprocess")

    with pytest.raises(ImportError):
        safe_imp("shutil")

    with pytest.raises(ImportError):
        safe_imp("socket")

    with pytest.raises(ImportError):
        safe_imp("requests")


def test_has_while_loop_detection():
    assert has_while_loop("while True: pass")
    assert has_while_loop("i = 0\nwhile i < 10:\n    i += 1")
    assert not has_while_loop("for i in range(10): print(i)")
    assert not has_while_loop("a = 1 + 2\nresult = a")


def test_extract_python_code_markdown():
    # Khối markdown chuẩn
    res1 = extract_python_code("Lời giải:\n```python\nresult = {'type': 'number', 'answer': 100}\n```")
    assert "result = {'type': 'number', 'answer': 100}" in res1

    # Khối không có thẻ ngôn ngữ
    res2 = extract_python_code("```\nresult = 42\n```")
    assert "result = 42" in res2

    # Code trần không có backticks
    res3 = extract_python_code("result = {'type': 'number', 'answer': 99}")
    assert "result = {'type': 'number', 'answer': 99}" in res3


def test_safe_duckdb_wrapper_backtick_and_column_defense():
    con = duckdb.connect()
    df = pd.DataFrame({"Thành tiền": [100.0, 200.0], "Số lượng": [2, 5]})
    con.register("hoa_don", df)
    tables = {"hoa_don": df}

    safe_con = SafeDuckDBWrapper(con, tables=tables)

    # 1. Tự động sửa backticks MySQL thành double quotes
    res_tick = safe_con.execute("SELECT `Thành tiền` FROM hoa_don").fetchall()
    assert len(res_tick) == 2

    # 2. Phòng thủ cột giả định: status_norm không có trong bảng
    res_status = safe_con.execute("SELECT * FROM hoa_don WHERE \"status_norm\" = 'Hoàn thành'").fetchall()
    assert len(res_status) == 2

    # 3. Phòng thủ cột giả định: revenue_norm fallback sang Thành tiền
    res_rev = safe_con.execute("SELECT SUM(COALESCE(\"revenue_norm\", \"Thành tiền\")) FROM hoa_don").fetchall()
    assert res_rev[0][0] == 300.0

    # 4. Phòng thủ cột giả định: qty_base fallback sang Số lượng
    res_qty = safe_con.execute("SELECT SUM(COALESCE(\"qty_base\", \"Số lượng\")) FROM hoa_don").fetchall()
    assert res_qty[0][0] == 7.0


def test_safe_duckdb_wrapper_multiple_ctes():
    con = duckdb.connect()
    df = pd.DataFrame({"Thành tiền": [100.0, 200.0, 300.0], "Mã đơn": ["A", "B", "C"]})
    con.register("don_hang", df)
    safe_con = SafeDuckDBWrapper(con, tables={"don_hang": df})

    sql = """
    WITH cte1 AS (
        SELECT "Mã đơn", "Thành tiền" * 2 AS tt_double FROM don_hang
    ),
    cte2 AS (
        SELECT SUM(tt_double) AS tong_gap_doi FROM cte1
    )
    SELECT tong_gap_doi FROM cte2
    """
    res = safe_con.execute(sql).fetchall()
    assert res[0][0] == 1200.0


def test_execute_sandboxed_code_clean_in_process():
    from src.sandbox import Sandbox
    sb = Sandbox(data_root=Path("/Applications/pipeline_datamate/assets"))
    con = duckdb.connect()
    df = pd.DataFrame({"Thành tiền": [15, 25, 60]})
    con.register("don_hang", df)
    sb._warm_store.create_duckdb_connection = lambda bid: (con, {"don_hang": df})

    code = "result = {'type': 'number', 'answer': int(tables['don_hang']['Thành tiền'].sum())}"
    res = sb._execute_in_process(code, business_id="biz_01", as_of_date="2026-05-15")
    assert res["ok"] is True
    assert res["result"] == {"type": "number", "answer": 100}


def test_execute_sandboxed_code_blocks_malicious_open():
    from src.sandbox import Sandbox
    sb = Sandbox(data_root=Path("/Applications/pipeline_datamate/assets"))
    con = duckdb.connect()
    sb._warm_store.create_duckdb_connection = lambda bid: (con, {})

    code = "f = open('/tmp/hacked.txt', 'w')\nf.write('malicious')"
    res = sb._execute_in_process(code, business_id="biz_01", as_of_date="2026-05-15")
    assert res["ok"] is False
    assert "name 'open' is not defined" in res["error"] or "ImportError" in res["error"]


def test_execute_sandboxed_code_syntax_error():
    from src.sandbox import Sandbox
    sb = Sandbox(data_root=Path("/Applications/pipeline_datamate/assets"))
    code = "def invalid_syntax(:"
    res = sb.execute(code, business_id="biz_01", as_of_date="2026-05-15")
    assert res["ok"] is False
    assert "SyntaxError" in res["error"]


def test_safe_builtins_blocks_system_inspection():
    safe_import = make_safe_import(duckdb, pd, np)
    builtins_dict = make_safe_builtins(safe_import)

    # Chặn các built-in nguy hiểm
    for dangerous in ["breakpoint", "help", "input", "memoryview"]:
        assert dangerous not in builtins_dict
