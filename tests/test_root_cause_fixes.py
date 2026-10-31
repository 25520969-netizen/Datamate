# -*- coding: utf-8 -*-
"""
Bộ kiểm thử xác nhận các bản sửa lỗi gốc rễ (Root Cause Bug Fixes):
1. Cột `business_id` được tự động nạp vào mọi bảng và DuckDB truy vấn `WHERE business_id = '...'` không còn lỗi Binder Error.
2. Column synonym aliases (Tên khách <-> Tên KH, Mã đơn hàng <-> Mã chứng từ, Trạng thái <-> status_norm).
3. DuckDB `strip_accents` xử lý tìm kiếm tiếng Việt không dấu / có dấu.
4. `DynamicEntityGrounder` khớp cụm từ n-gram con (subphrase) cho câu hỏi viết rút gọn.
5. `postprocess_and_guard` kích hoạt Sanity Guard khi câu hỏi tỉ lệ đơn hủy hoặc số lượng trả về 0 / 0.0%.
6. Cơ chế Arbitration từ chối fallback đơn độc của Prompt 2 (p2_only_rejected_abstain) để bảo vệ điểm phạt.
"""

from pathlib import Path
from typing import Any, Dict, List
import duckdb
import pandas as pd
import pytest

from model.baseline import DataMateAgent, QwenCoderClient
from src.ingest import BusinessDataStore
from src.postprocess import postprocess_and_guard, check_consensus, are_strings_in_consensus
from src.rule_engine import DynamicEntityGrounder, strip_accents
from src.sandbox import PythonSandbox, has_while_loop


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "train" / "data"
CALENDAR_PATH = PROJECT_ROOT / "train" / "calendar.csv"
RULEBOOK_PATH = PROJECT_ROOT / "train" / "rulebook.md"


def test_business_id_and_column_synonyms_in_duckdb(tmp_path: Path):
    """Kiểm tra cột business_id và synonym aliases được nạp và hoạt động trong DuckDB."""
    store = BusinessDataStore(data_root=DATA_ROOT, cache_root=tmp_path / "cache")
    con, tables = store.create_duckdb_connection("biz_101")

    for tbl_name, df in tables.items():
        assert "business_id" in df.columns, f"Bảng {tbl_name} thiếu cột business_id"
        assert df["business_id"].iloc[0] == "biz_101"

    # Kiểm tra truy vấn WHERE business_id = 'biz_101' không ném Binder Error
    first_tbl = list(tables.keys())[0]
    res = con.execute(f'SELECT COUNT(*) FROM "{first_tbl}" WHERE "business_id" = \'biz_101\'').fetchone()
    assert res[0] > 0

    # Kiểm tra column aliases nếu bảng có cột thời gian/ngày
    if "Ngày" in tables[first_tbl].columns:
        res_date = con.execute(f'SELECT "Thời gian" FROM "{first_tbl}" LIMIT 1').fetchone()
        assert res_date is not None


def test_duckdb_strip_accents_vietnamese():
    """Kiểm tra hàm DuckDB strip_accents xử lý chuẩn xác chuỗi tiếng Việt có dấu và không dấu."""
    con = duckdb.connect(database=":memory:")
    df = pd.DataFrame({
        "Tên sản phẩm": [
            "Gạo nếp cái hoa vàng (kg)",
            "Lẩu gà lá é",
            "Khô mực nướng",
            "Cơm chiên hải sản",
        ],
        "Số lượng": [32, 137, 50, 172],
    })
    con.register("san_pham", df)

    # 1. 'nep cai hoa vang' khớp 'Gạo nếp cái hoa vàng (kg)'
    q1 = """
    SELECT SUM("Số lượng") FROM san_pham 
    WHERE strip_accents("Tên sản phẩm") ILIKE strip_accents('%nep cai hoa vang%')
    """
    res1 = con.execute(q1).fetchone()[0]
    assert res1 == 32

    # 2. 'lau ga la e' khớp 'Lẩu gà lá é'
    q2 = """
    SELECT SUM("Số lượng") FROM san_pham 
    WHERE strip_accents("Tên sản phẩm") ILIKE strip_accents('%lau ga la e%')
    """
    res2 = con.execute(q2).fetchone()[0]
    assert res2 == 137


def test_dynamic_entity_grounder_subphrase():
    """Kiểm tra DynamicEntityGrounder tìm đúng thực thể khi câu hỏi dùng cụm từ con không dấu."""
    grounder = DynamicEntityGrounder()
    mock_tables = {
        "don_hang": pd.DataFrame({
            "Tên hàng hóa": ["Gạo nếp cái hoa vàng (kg)", "Lẩu gà lá é", "Trà đào cam sả"],
            "Đơn vị tính": ["kg", "nồi", "ly"],
        })
    }

    q = "soos kg nếp cái hoa vàng bán đc trong tháng 9/2025 là bao nhiêu"
    hints = grounder.find_grounding_hints(q, "biz_test", mock_tables)
    assert "Gạo nếp cái hoa vàng (kg)" in hints
    assert "kg" in hints

    q2 = "trong thang 1 nam 2025 cua hang ban duoc may noi lau ga la e"
    hints2 = grounder.find_grounding_hints(q2, "biz_test", mock_tables)
    assert "Lẩu gà lá é" in hints2
    assert "nồi" in hints2


def test_sanity_guards_for_zero_answers():
    """Kiểm tra Sanity Guard bảo vệ không đoán 0.0% hay 0 cho các câu hỏi tỷ lệ hủy và số lượng."""
    # 1. Tỉ lệ đơn hủy = 0.0% -> abstain
    res_cancel = postprocess_and_guard(
        question_id="q_cancel_test",
        question="Trong tháng này, tỉ lệ đơn huỷ là bao nhiêu phần trăm?",
        raw_output={"type": "percent", "answer": 0.0},
        expected_type="percent",
    )
    assert res_cancel["type"] == "abstain"

    # 2. Tỉ trọng đóng góp doanh thu = 0.0% -> abstain
    res_share = postprocess_and_guard(
        question_id="q_share_test",
        question="doanh thu nhom hang dien chiem bao nhieu % tong doanh thu",
        raw_output={"type": "percent", "answer": 0.0},
        expected_type="percent",
    )
    assert res_share["type"] == "abstain"

    # 3. Số nồi/kg sản phẩm = 0 -> abstain
    res_qty = postprocess_and_guard(
        question_id="q_qty_test",
        question="trong thang 1 nam 2025 cua hang ban duoc may noi lau ga la e",
        raw_output={"type": "number", "answer": 0},
        expected_type="number",
    )
    assert res_qty["type"] == "abstain"

    # 4. Số lượng có giá trị thật > 0 -> giữ nguyên đáp án
    res_valid = postprocess_and_guard(
        question_id="q_valid_test",
        question="trong thang 1 nam 2025 cua hang ban duoc may noi lau ga la e",
        raw_output={"type": "number", "answer": 137},
        expected_type="number",
    )
    assert res_valid["type"] == "number"
    assert res_valid["answer"] == 137


def test_arbitration_rejects_p2_fallback_to_protect_score(tmp_path: Path):
    """
    Kiểm tra cơ chế Arbitration: Khi Prompt 1 lỗi, từ chối nhận Prompt 2 đơn độc (p2_only_rejected_abstain),
    vì dữ liệu thực nghiệm chứng minh Prompt 2 đơn độc chỉ đạt 29.5% chính xác (làm mất 18 điểm).
    """
    data_root = PROJECT_ROOT / "train_preprocessed_full"
    if not data_root.exists():
        data_root = PROJECT_ROOT / "train" / "data"

    def mock_p1_fails_p2_succeeds(messages: List[Dict[str, str]]) -> str:
        sys_msg = messages[0]["content"]
        if "SQL" in sys_msg:
            # Prompt 1 ném lỗi cú pháp/logic
            return "```python\nraise ValueError('Prompt 1 SQL failed')\n```"
        else:
            # Prompt 2 sinh đáp án
            return '```python\nresult = {"type": "number", "answer": 99999999}\n```'

    client = QwenCoderClient(backend="custom", custom_generator=mock_p1_fails_p2_succeeds)
    agent = DataMateAgent(
        data_root=data_root,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        consensus_mode="dual",
        arbitration_strategy="adaptive",
    )

    q_obj = {
        "id": "q_p2_reject_test",
        "business_id": "biz_120",
        "question": "Tổng doanh thu năm 2025 là bao nhiêu?",
        "as_of_date": "2025-04-30",
    }
    pred, details = agent.answer_question_dual(q_obj)
    assert pred["type"] == "abstain"
    assert details["decision_reason"] == "p2_only_rejected_abstain"


def test_sandbox_while_loop_routed_to_subprocess_and_terminates(tmp_path: Path):
    """
    Kiểm tra Lưu ý 1: Vòng lặp while được AST nhận diện tự động và điều hướng sang subprocess.
    Khi chạy mã vòng lặp vô tận, sandbox subprocess chấm dứt an toàn qua timeout,
    không để lại zombie thread chiếm CPU in-process.
    """
    code_while = """
import time
while True:
    time.sleep(0.01)
"""
    assert has_while_loop(code_while) is True
    assert has_while_loop("for i in range(10): pass") is False
    assert has_while_loop("result = con.execute('SELECT 1').df()") is False

    data_root = PROJECT_ROOT / "train_preprocessed_full"
    if not data_root.exists():
        data_root = PROJECT_ROOT / "train" / "data"

    sandbox = PythonSandbox(
        data_root=data_root,
        cache_root=tmp_path / "cache",
        timeout_seconds=1.0,
        in_process=True,
    )
    # Thực thi mã có while True -> sandbox tự phát hiện và chạy qua subprocess với timeout 1.0s
    res = sandbox.execute(
        code=code_while,
        business_id="biz_101",
        as_of_date="2025-04-30",
    )
    assert res["ok"] is False
    assert "TimeoutError" in str(res["error"])


def test_sandbox_in_process_duckdb_interrupt_on_timeout(tmp_path: Path):
    """
    Kiểm tra Lưu ý 1: Trong chế độ in-process, nếu truy vấn DuckDB C++ bị treo quá timeout,
    hàm raw_con.interrupt() và inject ngoại lệ SystemExit kích hoạt giải phóng tài nguyên ngay lập tức.
    """
    data_root = PROJECT_ROOT / "train_preprocessed_full"
    if not data_root.exists():
        data_root = PROJECT_ROOT / "train" / "data"

    sandbox = PythonSandbox(
        data_root=data_root,
        cache_root=tmp_path / "cache",
        timeout_seconds=0.5,
        in_process=True,
    )
    # Truy vấn DuckDB tính toán rất nặng trên bảng lớn gây timeout
    heavy_code = """
import duckdb
result = con.execute("SELECT count(*) FROM (SELECT range(1000000000))").df()
"""
    res = sandbox.execute(
        code=heavy_code,
        business_id="biz_101",
        as_of_date="2025-04-30",
    )
    assert res["ok"] is False
    assert "TimeoutError" in str(res["error"])


def test_consensus_semantic_token_difference_product_variants():
    """
    Kiểm tra Lưu ý 2: Thuật toán so khớp an toàn ngữ nghĩa (Semantic Token Difference).
    - Đồng thuận với các biến thể chỉ khác biệt về đơn vị đo lường hoặc tiền tố ngành hàng chung.
    - TUYỆT ĐỐI KHÔNG ĐỒNG THUẬN (trả về False -> Abstain an toàn) với các biến thể sản phẩm khác nhau (Omo vs Omo Matic).
    """
    # 1. Các trường hợp biến thể sản phẩm khác nhau -> BẮT BUỘC BÁC BỎ (False)
    assert are_strings_in_consensus("Bột giặt Omo", "Bột giặt Omo Matic") is False
    assert are_strings_in_consensus("Bia Tiger", "Bia Tiger Crystal") is False
    assert are_strings_in_consensus("Heineken", "Heineken Silver") is False
    assert are_strings_in_consensus("Panadol", "Panadol Extra") is False
    assert are_strings_in_consensus("Coca Cola", "Coca Cola Zero") is False
    assert are_strings_in_consensus("Bia", "Bia lon") is False  # Không có từ nội dung riêng

    # 2. Các trường hợp đúng cùng sản phẩm nhưng khác cách gọi đơn vị / tiền tố -> ĐỒNG THUẬN (True)
    assert are_strings_in_consensus("Bia Tiger lon (lon)", "Bia Tiger lon") is True
    assert are_strings_in_consensus("Bia Heineken lon", "Heineken lon") is True
    assert are_strings_in_consensus("Bia Tiger", "Tiger") is True
    assert are_strings_in_consensus("Gạo nếp cái hoa vàng (kg)", "Gạo nếp cái hoa vàng") is True
    assert are_strings_in_consensus("Gạo nếp cái hoa vàng kg", "Gạo nếp cái hoa vàng") is True

    # 3. Kiểm tra qua hàm check_consensus cấp cao cho kiểu string
    assert check_consensus(
        {"type": "string", "answer": "Bột giặt Omo"},
        {"type": "string", "answer": "Bột giặt Omo Matic"},
    ) is False

    assert check_consensus(
        {"type": "string", "answer": "Bia Tiger lon"},
        {"type": "string", "answer": "Bia Tiger"},
    ) is True

    # 4. Kiểm tra qua hàm check_consensus cấp cao cho kiểu list
    assert check_consensus(
        {"type": "list", "answer": ["Bia Tiger lon", "Heineken"]},
        {"type": "list", "answer": ["Bia Tiger", "Bia Heineken"]},
    ) is True

    assert check_consensus(
        {"type": "list", "answer": ["Bột giặt Omo"]},
        {"type": "list", "answer": ["Bột giặt Omo Matic"]},
    ) is False
