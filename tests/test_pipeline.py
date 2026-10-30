# -*- coding: utf-8 -*-
"""
Bộ kiểm thử toàn diện (Unit & Integration Tests) cho cả 5 Module của DataMate Pipeline:
  - Module 1: `src/ingest.py` (CSV, Excel nhiều sheet theo tháng, Excel đa bảng, SQLite, Rulebook §8)
  - Module 2: `src/date_resolver.py` & `src/rule_engine.py` (Lịch âm Tết, tháng/tuần tương đối, trích xuất luật)
  - Module 3: `model/baseline.py` (DataMateAgent với Qwen2.5-Coder-7B-Instruct & vòng lặp tự sửa lỗi)
  - Module 4: `src/sandbox.py` (Thực thi cô lập, bắt Traceback, chống treo Timeout)
  - Module 5: `src/postprocess.py` (Chuẩn hóa Mục 9 Rulebook & Penalized Accuracy Guard)
"""

from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluate import score_single_prediction
from model.baseline import DEFAULT_QWEN_7B_MODEL, DataMateAgent, QwenCoderClient
from src.date_resolver import DateResolver
from src.ingest import (
    BusinessDataStore,
    apply_rulebook_section8_cleaning,
    extract_year_from_sheet_name,
    normalize_date_value_iso,
    parse_vn_numeric_value,
    CANONICAL_STATUS_MAP,
    ORDER_STATUS_COL_NAMES,
)
from src.postprocess import postprocess_and_guard
from src.rule_engine import RuleEngine
from src.sandbox import PythonSandbox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
DATA_ROOT = TRAIN_DIR / "data"
CALENDAR_PATH = TRAIN_DIR / "calendar.csv"
RULEBOOK_PATH = TRAIN_DIR / "rulebook.md"


# ==============================================================================
# 1. KIỂM THỬ MODULE 2A: DATE RESOLVER
# ==============================================================================
def test_date_resolver_relative_and_lunar_tet():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)

    # Kiểm tra quy tắc "so với tháng trước" (cùng số ngày): 2026-03-31 -> tháng 2/2026 chỉ đến 28/02
    ctx_mar = resolver.resolve("doanh thu tháng này so với tháng trước", "2026-03-31")
    assert ctx_mar["this_month"] == ("2026-03-01", "2026-03-31")
    assert ctx_mar["tháng này"] == ("2026-03-01", "2026-03-31")
    assert ctx_mar["tuần này"] == ctx_mar["this_week"]
    assert ctx_mar["prev_month_same_days"] == ("2026-02-01", "2026-02-28")

    # Kiểm tra quy tắc "Tháng M không ghi năm": tại 2026-06-02, "tháng 12" là 12/2025, "tháng 5" là 05/2026
    ctx_dec = resolver.resolve("đếm số đơn đã huỷ trong tháng 12", "2026-06-02")
    periods_dec = {p["year_month"]: (p["start_date"], p["end_date"]) for p in ctx_dec["detected_periods"] if "year_month" in p}
    assert "2025-12" in periods_dec
    assert periods_dec["2025-12"] == ("2025-12-01", "2025-12-31")

    # Kiểm tra Tết năm 2025 và Trước Tết 2025 từ calendar.csv
    tet_2025 = resolver.get_tet_range(2025)
    pre_tet_2025 = resolver.get_pre_tet_range(2025)
    assert tet_2025 is not None
    assert pre_tet_2025 is not None
    tet_days = (tet_2025[1] - tet_2025[0]).days + 1
    pre_days = (pre_tet_2025[1] - pre_tet_2025[0]).days + 1
    assert tet_days == pre_days
    assert (tet_2025[0] - pre_tet_2025[1]).days == 1


# ==============================================================================
# 2. KIỂM THỬ MODULE 2B: RULE ENGINE
# ==============================================================================
def test_rule_engine_type_inference_and_knowledge_injection():
    engine = RuleEngine(rulebook_path=RULEBOOK_PATH)

    assert engine.infer_expected_type("tỉ lệ huỷ đơn tháng 6/2025 là nhiêu %") == "percent"
    assert engine.infer_expected_type("3 mặt hàng bán chạy nhất tuần này là những mặt hàng nào?") == "list"
    assert engine.infer_expected_type("thang 6/2025 hang nao chay nhat") == "string"
    assert engine.infer_expected_type("Tổng doanh thu trong ngày hôm nay là bao nhiêu?") == "number"
    assert engine.infer_expected_type("Lập bảng 5 khách hàng còn nợ nhiều nhất kèm số tiền nợ") == "table"

    rules_pharmacy = engine.select_relevant_rules("tỉ lệ thuốc kê đơn trong tháng 7/2025")
    assert "Nhà thuốc" in rules_pharmacy
    assert "Thuốc kê đơn" in rules_pharmacy


# ==============================================================================
# 3. KIỂM THỬ MODULE 1: INGESTION & RULEBOOK SECTION 8 CLEANING
# ==============================================================================
def test_rulebook_section8_filldown_and_dedup():
    # Bảng có cột Mã đơn hàng: phải fill-down thông tin cấp đơn và xóa dòng trùng hệt nhau trong cùng 1 đơn
    df_with_order = pd.DataFrame(
        [
            {"Mã đơn hàng": "DH01", "Ngày đặt hàng": "01/05/2025", "Trạng thái đơn hàng": "Hoàn thành", "Tên sản phẩm": "SP A", "Tổng tiền": 10000},
            {"Mã đơn hàng": "", "Ngày đặt hàng": "", "Trạng thái đơn hàng": "", "Tên sản phẩm": "SP B", "Tổng tiền": 20000},
            # Dòng trùng hệt dòng 1 của cùng đơn DH01 (lỗi xuất file -> phải bỏ)
            {"Mã đơn hàng": "DH01", "Ngày đặt hàng": "01/05/2025", "Trạng thái đơn hàng": "Hoàn thành", "Tên sản phẩm": "SP A", "Tổng tiền": 10000},
        ]
    )
    cleaned = apply_rulebook_section8_cleaning(df_with_order)
    assert len(cleaned) == 2
    assert cleaned.loc[1, "Mã đơn hàng"] == "DH01"
    assert cleaned.loc[1, "Ngày đặt hàng"] == "01/05/2025"
    assert cleaned.loc[1, "Trạng thái đơn hàng"] == "Hoàn thành"

    # Bảng KHÔNG có cột mã đơn (ví dụ biz_107, biz_113, biz_126): hai dòng giống nhau vẫn phải tính cả hai!
    df_no_order = pd.DataFrame(
        [
            {"ngay_ban": "01/05/2025", "Nội dung": "Coca Cola 330ml", "Tổng tiền": 10000},
            {"ngay_ban": "01/05/2025", "Nội dung": "Coca Cola 330ml", "Tổng tiền": 10000},
        ]
    )
    cleaned_no_order = apply_rulebook_section8_cleaning(df_no_order)
    assert len(cleaned_no_order) == 2

    # Kiểm tra chuẩn hóa ngày hỗn hợp ('1 thg 6 2025', Excel serial '45689') và năm sheet 2 chữ số
    assert normalize_date_value_iso("1 thg 6 2025") == "2025-06-01"
    assert normalize_date_value_iso("45689") == "2025-02-01"
    assert normalize_date_value_iso(45689) == "2025-02-01"
    assert extract_year_from_sheet_name("T01-25") == 2025
    assert extract_year_from_sheet_name("Tháng 02/2026") == 2026


def test_ingest_real_businesses_csv_excel_sqlite(tmp_path: Path):
    store = BusinessDataStore(data_root=DATA_ROOT, cache_root=tmp_path / "cache")

    # 1. CSV phẳng (biz_101)
    t_101 = store.load_tables("biz_101")
    assert "ban_hang" in t_101
    assert len(t_101["ban_hang"]) > 0

    # 2. Excel đa bảng nằm chung 1 file DuLieuBanHang.xlsx (biz_109)
    t_109 = store.load_tables("biz_109")
    assert "DanhSachHoaDon" in t_109
    assert "DanhSachSanPham" in t_109
    assert "DanhSachHoaDonChiTiet" in t_109

    # 3. SQLite 17 bảng (biz_123)
    t_123 = store.load_tables("biz_123")
    assert "don_hang" in t_123
    assert "chi_tiet_don" in t_123
    assert "thong_tin_thuoc" in t_123


# ==============================================================================
# 4. KIỂM THỬ MODULE 4 & 5: SANDBOX & POST-PROCESSING
# ==============================================================================
def test_sandbox_and_postprocess_real_question_q_002d03ebbded(tmp_path: Path):
    """
    Kiểm chứng thực tế câu q_002d03ebbded trên biz_120:
    Câu hỏi: 'Tổng doanh thu trong ngày hôm nay là bao nhiêu?', as_of_date = '2025-04-30'
    Đáp án chuẩn trong answers.jsonl: 10309400
    """
    sandbox = PythonSandbox(data_root=DATA_ROOT, cache_root=tmp_path / "cache", timeout_seconds=10.0)

    code = """
hd = tables["DanhSachHoaDon"].copy()
ct = tables["DanhSachHoaDonChiTiet"].copy()
lo = tables["DanhSachHoaDonChiTietLo"].copy()

# Lọc đơn ngày 30/04/2025, không huỷ
hd["ngay_str"] = hd["Thời gian"].astype(str).str.slice(0, 10)
valid_hd = hd[
    (hd["ngay_str"].isin(["2025-04-30", "30/04/2025"]))
    & (~hd["Trạng thái"].astype(str).str.lower().str.contains("huỷ|hủy|trả", regex=True))
]

# Loại bỏ đơn có trả hàng (Số lượng trả > 0 trong DanhSachHoaDonChiTietLo)
returned_orders = set(
    lo[pd.to_numeric(lo["Số lượng trả (ĐVT cơ bản)"], errors="coerce").fillna(0) > 0]["Mã hóa đơn"]
)
valid_order_ids = set(valid_hd["Mã hóa đơn"]) - returned_orders

# Doanh thu = tổng Thành tiền các dòng hàng của đơn hợp lệ
rev = ct[ct["Mã hóa đơn"].isin(valid_order_ids)]["Thành tiền"].sum()
result = {"type": "number", "answer": float(rev)}
"""
    exec_out = sandbox.execute(code=code, business_id="biz_120", as_of_date="2025-04-30")
    assert exec_out["ok"] is True, f"Sandbox error: {exec_out.get('error')}"

    final_pred = postprocess_and_guard(
        question_id="q_002d03ebbded",
        question="Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        raw_output=exec_out["result"],
        expected_type="number",
    )
    gold_obj = {"id": "q_002d03ebbded", "type": "number", "answer": 10309400, "alias": []}
    assert score_single_prediction(final_pred, gold_obj) == (1, "correct")


def test_postprocess_penalized_accuracy_guard():
    # Doanh thu âm vô lý -> phải kích hoạt abstain (0 điểm thay vì -1)
    pred = postprocess_and_guard(
        question_id="q_neg",
        question="Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        raw_output={"type": "number", "answer": -500000},
        expected_type="number",
    )
    assert pred == {"id": "q_neg", "type": "abstain"}

    # Phần trăm < -100% vô lý -> phải kích hoạt abstain
    pred_pct = postprocess_and_guard(
        question_id="q_pct",
        question="tỉ lệ huỷ đơn tháng 6/2025 là nhiêu %",
        raw_output={"type": "percent", "answer": -150.5},
        expected_type="percent",
    )
    assert pred_pct == {"id": "q_pct", "type": "abstain"}


def test_strip_trailing_measurement_and_percent_scaling():
    from src.postprocess import strip_trailing_measurement_in_parens

    # Giữ nguyên hậu tố kích cỡ hoặc quy cách sản phẩm
    assert strip_trailing_measurement_in_parens("Cà phê sữa (M)") == "Cà phê sữa (M)"
    assert strip_trailing_measurement_in_parens("Trà đào cam sả (L)") == "Trà đào cam sả (L)"
    assert strip_trailing_measurement_in_parens("Panadol Extra (500mg)") == "Panadol Extra (500mg)"
    assert strip_trailing_measurement_in_parens("Khăn ướt (gói)") == "Khăn ướt (gói)"

    # Loại bỏ tiền hoặc số lượng kèm đơn vị do LLM tự chú thích
    assert strip_trailing_measurement_in_parens("Nguyễn Văn A (1.500.000đ)") == "Nguyễn Văn A"
    assert strip_trailing_measurement_in_parens("Trà sữa trân châu (307 ly)") == "Trà sữa trân châu"
    assert strip_trailing_measurement_in_parens("Khách hàng KH0014 (1,605,000)") == "Khách hàng KH0014"

    # Tự động quy đổi tỷ số thập phân 0 < x < 1.0 sang % chuẩn
    res_pct = postprocess_and_guard(
        question_id="q_ratio",
        question="Tỉ lệ hủy đơn là bao nhiêu phần trăm?",
        raw_output={"type": "percent", "answer": 0.403},
        expected_type="percent",
    )
    assert res_pct["answer"] == 40.3

    res_pct2 = postprocess_and_guard(
        question_id="q_ratio2",
        question="Tỉ lệ là bao nhiêu?",
        raw_output={"type": "percent", "answer": 35.17},
        expected_type="percent",
    )
    assert res_pct2["answer"] == 35.17


# ==============================================================================
# 5. KIỂM THỬ MODULE 3: END-TO-END AGENT & SELF-CORRECTION LOOP (QWEN 7B)
# ==============================================================================
def test_datamate_agent_self_correction_and_abstention(tmp_path: Path):
    call_count = {"n": 0}

    def mock_qwen7b_generator(messages: List[Dict[str, str]]) -> str:
        call_count["n"] += 1
        # Lượt 1: Giả lập LLM viết sai tên cột gây KeyError
        if call_count["n"] == 1:
            return "```python\ndf = tables['DanhSachHoaDon']\nresult = df['Cot_Khong_Ton_Tai'].sum()\n```"
        # Lượt 2: Nhận được Traceback từ Sandbox, LLM tự sửa lại code đúng
        assert "KeyError" in messages[-1]["content"]
        return """```python
hd = tables["DanhSachHoaDon"].copy()
ct = tables["DanhSachHoaDonChiTiet"].copy()
lo = tables["DanhSachHoaDonChiTietLo"].copy()
hd["ngay_str"] = hd["Thời gian"].astype(str).str.slice(0, 10)
valid_hd = hd[
    (hd["ngay_str"].isin(["2025-04-30", "30/04/2025"]))
    & (~hd["Trạng thái"].astype(str).str.lower().str.contains("huỷ|hủy|trả", regex=True))
]
returned_orders = set(
    lo[pd.to_numeric(lo["Số lượng trả (ĐVT cơ bản)"], errors="coerce").fillna(0) > 0]["Mã hóa đơn"]
)
valid_ids = set(valid_hd["Mã hóa đơn"]) - returned_orders
rev = ct[ct["Mã hóa đơn"].isin(valid_ids)]["Thành tiền"].sum()
result = {"type": "number", "answer": float(rev)}
```"""

    client = QwenCoderClient(backend="custom", custom_generator=mock_qwen7b_generator)
    assert client.model_name == DEFAULT_QWEN_7B_MODEL

    agent = DataMateAgent(
        data_root=DATA_ROOT,
        calendar_path=CALENDAR_PATH,
        rulebook_path=RULEBOOK_PATH,
        llm_client=client,
        cache_root=tmp_path / "cache",
        max_retries=2,
    )

    q_obj = {
        "id": "q_002d03ebbded",
        "business_id": "biz_120",
        "question": "Tổng doanh thu trong ngày hôm nay là bao nhiêu?",
        "as_of_date": "2025-04-30",
    }
    pred = agent.answer_question(q_obj)
    assert call_count["n"] == 2
    assert pred == {"id": "q_002d03ebbded", "type": "number", "answer": 10309400}


def test_space_number_parsing_and_status_canonicalization():
    # 1. Kiểm tra số có dấu cách phân cách hàng nghìn
    assert parse_vn_numeric_value("129 000") == 129000.0
    assert parse_vn_numeric_value("10 000") == 10000.0
    assert parse_vn_numeric_value("16 076") == 16076.0

    # 2. Kiểm tra chuẩn hóa trạng thái và các cột trạng thái mới
    assert CANONICAL_STATUS_MAP.get("khách trả") == "Trả hàng"
    assert CANONICAL_STATUS_MAP.get("đang ship") == "Đang giao"
    assert "trạng thái giao" in ORDER_STATUS_COL_NAMES
    assert "tt đơn" in ORDER_STATUS_COL_NAMES

    # 3. Kiểm tra lọc dòng lặp lại header phân trang (pagination header row)
    df_dirty = pd.DataFrame([
        {"Mã": "M1", "Tên": "SP A", "Tiền": "1000"},
        {"Mã": "Mã", "Tên": "Tên", "Tiền": "Tiền"},
        {"Mã": "M2", "Tên": "SP B", "Tiền": "2000"},
    ])
    cleaned = apply_rulebook_section8_cleaning(df_dirty)
    assert len(cleaned) == 2
    assert cleaned["Mã"].tolist() == ["M1", "M2"]


def test_improved_postprocess_and_date_dialects():
    # 1. Date resolver phương ngữ
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    ctx_hom_ni = resolver.resolve("Hom ni quan duoc may don roi?", "2025-09-09")
    periods = [p["note"] for p in ctx_hom_ni["detected_periods"]]
    assert any("2025-09-09" in p for p in periods)

    ctx_tuan_roi = resolver.resolve("Tuần rồi dthu là bnhiu?", "2025-09-09")
    p_tr = [p["expr"] for p in ctx_tuan_roi["detected_periods"]]
    assert any("Tuần trước" in p for p in p_tr)

    # 2. RuleEngine infer expected type
    re_eng = RuleEngine(rulebook_path=RULEBOOK_PATH)
    assert re_eng.infer_expected_type("Ai là khách có số đơn mua nhiều nhất?") == "string"
    assert re_eng.infer_expected_type("Top 3 khách hàng có công nợ lớn nhất kèm số tiền nợ?") == "table"

    # 3. Postprocess clean comma-separated list
    p_list = postprocess_and_guard(
        question_id="q1",
        question="Top 3 mặt hàng bán chạy nhất?",
        raw_output="Trà đào, Trà sữa, Bạc xỉu",
        expected_type="list",
    )
    assert p_list["type"] == "list"
    assert p_list["answer"] == ["Trà đào", "Trà sữa", "Bạc xỉu"]

    # 4. Postprocess strip parenthesized amount
    p_paren = postprocess_and_guard(
        question_id="q2",
        question="Ai là khách hàng có doanh thu cao nhất?",
        raw_output="KH0014 (1,605,000)",
        expected_type="string",
    )
    assert p_paren["type"] == "string"
    assert p_paren["answer"] == "KH0014"

    # 5. Postprocess abstain on Khách lẻ
    p_kl = postprocess_and_guard(
        question_id="q3",
        question="Khách nào mua nhiều tiền nhất?",
        raw_output="Khách lẻ",
        expected_type="string",
    )
    assert p_kl["type"] == "abstain"

    # 6. Postprocess abstain on 0 revenue today
    p_zero = postprocess_and_guard(
        question_id="q4",
        question="Hom ni quan duoc may don roi?",
        raw_output=0,
        expected_type="number",
    )
    assert p_zero["type"] == "abstain"


def test_subtotal_isolation_and_dynamic_entity_grounder():
    from src.rule_engine import DynamicEntityGrounder, strip_accents

    # 1. Kiểm tra cô lập dòng tổng kết (subtotal) không làm sai lệch đơn hàng
    df_with_subtotal = pd.DataFrame([
        {"Mã đơn": "DH01", "Ngày": "01/05/2025", "Status": "Hoàn thành", "Tên": "Cơm sườn", "Tiền": 50000},
        {"Mã đơn": "", "Ngày": "", "Status": "", "Tên": "Trà đá", "Tiền": 5000},
        {"Mã đơn": "", "Ngày": "01/05/2025", "Status": "", "Tên": "Cộng ngày 01/05/2025", "Tiền": 55000},
        {"Mã đơn": "DH02", "Ngày": "01/05/2025", "Status": "Hoàn thành", "Tên": "Phở bò", "Tiền": 45000},
    ])
    cleaned = apply_rulebook_section8_cleaning(df_with_subtotal)
    assert len(cleaned) == 4
    # Dòng 0 và 1 thuộc đơn DH01
    assert cleaned.loc[0, "Mã đơn"] == "DH01"
    assert cleaned.loc[1, "Mã đơn"] == "DH01"
    assert cleaned.loc[1, "Status"] == "Hoàn thành"
    # Dòng 2 là Subtotal: Mã đơn giữ nguyên NA, Status là 'Subtotal'
    assert pd.isna(cleaned.loc[2, "Mã đơn"])
    assert cleaned.loc[2, "Status"] == "Subtotal"
    # Dòng 3 thuộc đơn DH02
    assert cleaned.loc[3, "Mã đơn"] == "DH02"

    # Lọc đơn hoàn thành chỉ ra đúng 3 món (tổng 100k), không bị cộng thừa dòng Subtotal 55k!
    valid_orders = cleaned[cleaned["Status"] == "Hoàn thành"]
    assert valid_orders["Tiền"].sum() == 100000
    assert valid_orders["Mã đơn"].nunique() == 2

    # 2. Kiểm tra DynamicEntityGrounder
    grounder = DynamicEntityGrounder()
    mock_tables = {
        "thuc_don": pd.DataFrame([
            {"Tên món": "Cơm chiên hải sản", "ĐVT": "đĩa"},
            {"Tên món": "Lẩu hải sản", "ĐVT": "nồi"},
            {"Tên món": "Trà đào cam sả", "ĐVT": "ly"},
        ])
    }
    # Tìm kiếm không dấu
    hint1 = grounder.find_grounding_hints(
        question="so dia Com chien hai san ban duoc trong thang 4 la bao nhieu",
        business_id="test_shop",
        tables=mock_tables,
    )
    assert "Cơm chiên hải sản" in hint1
    assert "ĐVT: 'đĩa'" in hint1

    # Kiểm tra strip_accents
    assert strip_accents("Cơm chiên hải sản") == "com chien hai san"
    assert strip_accents("Đỗ Đức An") == "do duc an"


def test_hqua_date_resolution_and_unanswerable_inventory():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    ctx = resolver.resolve("Tổng dthu hqua là bao nhiêu?", "2026-02-10")
    detected = ctx.get("detected_periods", [])
    assert len(detected) >= 1
    assert detected[0]["start_date"] == "2026-02-09"
    assert detected[0]["end_date"] == "2026-02-09"

    engine = RuleEngine(rulebook_path=RULEBOOK_PATH)
    # Kiểm tra suy đoán type table
    assert engine.infer_expected_type("Top 3 khách hàng có công nợ lớn nhất đến hết ngày 28/07/2026 và số nợ của từng người?") == "table"
    assert engine.infer_expected_type("tính đến hết ngày 10/08/2025, 3 khách nào đang nợ nhìu nhất, mỗi người còn nợ bao nhiêu") == "table"

    # Kiểm tra unanswerable khi hỏi tồn kho trên shop không có bảng tồn kho / nhập hàng
    store = BusinessDataStore(data_root=DATA_ROOT)
    is_unans, reason = engine.detect_unanswerable_deterministic(
        question_obj={
            "id": "q_test_stock",
            "business_id": "biz_118",
            "question": "hết hôm nay, gạch thẻ đặc 4x8x18 còn laij mấy viên trong kho",
            "as_of_date": "2026-09-10",
        },
        store=store,
        date_resolver=resolver,
    )
    assert is_unans is True
    assert "tồn kho" in reason.lower()
