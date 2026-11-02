# -*- coding: utf-8 -*-
"""
Unit tests cho Module 1: Ingestion & Schema Standardization Engine.
Kiểm tra làm sạch Section 8 (subtotal row removal, fill-down, deduplication, unpivot matrix).
"""

import pandas as pd
import pytest
from src.ingest import (
    to_nfc,
    apply_rulebook_section8_cleaning,
    BusinessDataStore,
    normalize_date_value_iso,
)


def test_to_nfc_bom_and_spaces():
    raw_str = "\ufeff  Sản phẩm   A  "
    assert to_nfc(raw_str) == "Sản phẩm A"


def test_section8_subtotal_removal():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", "HD02", "HD03", None],
        "Tên hàng": ["Trà sữa", "Cà phê", "Bạc xỉu", "Tổng cộng"],
        "Thành tiền": [30000, 25000, 28000, 83000],
    })
    cleaned = apply_rulebook_section8_cleaning(df)
    # Dòng "Tổng cộng" phải bị khử hoàn toàn
    assert len(cleaned) == 3
    assert "Tổng cộng" not in cleaned["Tên hàng"].values


def test_section8_fill_down_order_level_columns():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", None, "HD02"],
        "Khách hàng": ["Anh Nam", None, "Chị Hoa"],
        "Trạng thái": ["Hoàn thành", None, "Đã hủy"],
        "Tên hàng": ["Món 1", "Món 2", "Món 3"],
    })
    cleaned = apply_rulebook_section8_cleaning(df)
    # Ô trống ở dòng thứ hai của cùng đơn HD01 phải được fill-down
    assert cleaned.loc[1, "Mã đơn"] == "HD01"
    assert cleaned.loc[1, "Khách hàng"] == "Anh Nam"
    assert cleaned.loc[1, "Trạng thái"] == "Hoàn thành"


def test_section8_deduplicate_identical_order_lines():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", "HD01", "HD02", "HD03"],
        "Tên hàng": ["Trà sữa", "Trà sữa", "Trà sữa", "Trà sữa"],
        "Số lượng": [1, 1, 1, 1],
        "Thành tiền": [30000, 30000, 30000, 30000],
    })
    cleaned = apply_rulebook_section8_cleaning(df)
    # 2 dòng của cùng đơn HD01 giống hệt nhau là lỗi xuất file -> chỉ tính 1 lần
    # 2 dòng giống nhau của hai đơn khác nhau HD02 và HD03 -> giữ cả hai
    assert len(cleaned) == 3


def test_matrix_unpivot_detection():
    df_matrix = pd.DataFrame({
        "Sản phẩm": ["Mặt hàng A", "Mặt hàng B"],
        "Tháng 1": [1000, 2000],
        "Tháng 2": [1500, 2500],
        "Tháng 3": [1800, 2800],
    })
    tables = {"bao_cao": df_matrix}
    res = BusinessDataStore._detect_and_unpivot_matrix_tables(tables)
    assert "bao_cao_unpivot" in res
    df_unp = res["bao_cao_unpivot"]
    assert "Thời gian" in df_unp.columns
    assert "Thành tiền" in df_unp.columns
    assert len(df_unp) == 6


def test_parse_vn_numeric_value_various_formats():
    from src.ingest import parse_vn_numeric_value
    # Dấu chấm hàng nghìn kèm đ/vnđ
    assert parse_vn_numeric_value("55.000đ") == 55000.0
    assert parse_vn_numeric_value("-53.500đ") == -53500.0
    # Dấu phẩy chuẩn quốc tế
    assert parse_vn_numeric_value("90,000") == 90000.0
    # Khoảng trắng phân cách
    assert parse_vn_numeric_value("129 000") == 129000.0
    # Viết tắt k (nghìn)
    assert parse_vn_numeric_value("25k") == 25000.0
    assert parse_vn_numeric_value("12.5k") == 12500.0
    # Viết tắt tr (triệu)
    assert parse_vn_numeric_value("1tr440") == 1440000.0
    assert parse_vn_numeric_value("2.5tr") == 2500000.0


def test_canonical_status_map_completeness():
    from src.ingest import CANONICAL_STATUS_MAP
    # Nhóm Hoàn thành
    assert CANONICAL_STATUS_MAP["thành công"] == "Hoàn thành"
    assert CANONICAL_STATUS_MAP["đã giao"] == "Hoàn thành"
    assert CANONICAL_STATUS_MAP["completed"] == "Hoàn thành"

    # Nhóm Đã hủy
    assert CANONICAL_STATUS_MAP["đã hủy"] == "Đã hủy"
    assert CANONICAL_STATUS_MAP["đã huỷ đơn"] == "Đã hủy"
    assert CANONICAL_STATUS_MAP["hủy"] == "Đã hủy"
    assert CANONICAL_STATUS_MAP["cancelled"] == "Đã hủy"

    # Nhóm Trả hàng
    assert CANONICAL_STATUS_MAP["trả hàng"] == "Trả hàng"
    assert CANONICAL_STATUS_MAP["hoàn trả"] == "Trả hàng"
    assert CANONICAL_STATUS_MAP["đổi trả"] == "Trả hàng"


def test_enrich_table_synonyms_adds_qty_base_and_status_norm():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", "HD02"],
        "Số lượng": [2, 5],
        "Trạng thái": ["Hoàn thành", "Đã huỷ đơn"],
        "Thành tiền": [20000, 50000],
    })
    tables = {"don_hang": df}
    enriched = BusinessDataStore._enrich_table_synonyms_and_id(tables, business_id="biz_01")
    df_res = enriched["don_hang"]

    assert "qty_base" in df_res.columns
    assert list(df_res["qty_base"]) == [2.0, 5.0]

    assert "status_norm" in df_res.columns
    assert df_res.loc[0, "status_norm"] == "Hoàn thành"
    assert df_res.loc[1, "status_norm"] == "Đã hủy"


def test_section8_subtotal_various_keywords():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", None, "HD02", None],
        "Tên hàng": ["Trà sữa", "Cộng tiền hàng", "Bánh", "Tổng cộng"],
        "Thành tiền": [30000, 30000, 20000, 20000],
    })
    cleaned = apply_rulebook_section8_cleaning(df)
    assert len(cleaned) == 2
    assert list(cleaned["Mã đơn"]) == ["HD01", "HD02"]


def test_fill_down_preserves_non_order_columns():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", None],
        "Tên hàng": ["Trà đào", "Trà vải"],
        "Đơn vị tính": ["Ly", None],
    })
    cleaned = apply_rulebook_section8_cleaning(df)
    # Mã đơn là order-level -> fill-down thành HD01
    assert cleaned.loc[1, "Mã đơn"] == "HD01"
    # Đơn vị tính là item-level -> không bị fill-down nhầm!
    assert pd.isna(cleaned.loc[1, "Đơn vị tính"])


def test_normalize_date_value_iso_and_auto_swap():
    # Chuẩn DD/MM/YYYY
    assert normalize_date_value_iso("25/12/2025") == "2025-12-25"
    # Hoán vị MM/DD/YYYY khi ngày > 12 nằm ở giữa
    assert normalize_date_value_iso("05/20/2025") == "2025-05-20"
    # Chuẩn ISO giữ nguyên
    assert normalize_date_value_iso("2026-03-15") == "2026-03-15"
    # Chuẩn có giờ phút giữ nguyên giờ để phân tích ca làm việc
    assert normalize_date_value_iso("2026-03-15 14:30:00") == "2026-03-15 14:30:00"
    # Giờ midnight (00:00:00) rút gọn về YYYY-MM-DD
    assert normalize_date_value_iso("2026-03-15 00:00:00") == "2026-03-15"


def test_enrich_table_synonyms_adds_date_iso():
    df = pd.DataFrame({
        "Mã đơn": ["HD01", "HD02"],
        "Ngày bán": ["15/05/2026", "20/05/2026"],
        "Thành tiền": [100000, 200000],
    })
    tables = {"ban_hang": df}
    enriched = BusinessDataStore._enrich_table_synonyms_and_id(tables, business_id="biz_test")
    res_df = enriched["ban_hang"]
    assert "date_iso" in res_df.columns
    assert list(res_df["date_iso"]) == ["2026-05-15", "2026-05-20"]
    assert "order_date_iso" in res_df.columns
    assert list(res_df["order_date_iso"]) == ["2026-05-15", "2026-05-20"]

