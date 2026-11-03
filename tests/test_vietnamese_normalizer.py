# -*- coding: utf-8 -*-
"""
Unit tests cho Module 1: Vietnamese Context-Aware Canonicalizer.
Kiểm tra chuẩn hoá Unicode, từ viết tắt và Bảo vệ Thực thể (Entity Preservation).
"""

import pandas as pd
import pytest
from src.vietnamese_normalizer import VietnameseNormalizer, to_nfc


def test_unicode_nfc_normalization():
    # Ký tự tổ hợp NFD vs NFC
    nfd_str = "th\u0061\u0301ng"  # tháng dạng NFD
    nfc_str = "tháng"
    assert to_nfc(nfd_str) == nfc_str


def test_no_hardcoded_typo_list():
    # Không còn danh sách lỗi gõ chép từ câu hỏi train: tầng tất định không tự "đoán" lỗi gõ,
    # để bước viết lại bằng LLM xử lý chung; tên thương hiệu không dấu không bị biến đổi.
    q = "naof thangs Tiger Box Max"
    assert VietnameseNormalizer.clean_deterministic(q) == q


def test_business_abbreviations_expansion():
    q = "tháng 5 dthu cua shop dc bnhiu, kh mua mấy sp có hsd đến tháng 6"
    cleaned = VietnameseNormalizer.clean_deterministic(q)
    assert "doanh thu" in cleaned
    assert "được" in cleaned
    assert "bao nhiêu" in cleaned
    assert "khách hàng" in cleaned
    assert "sản phẩm" in cleaned
    assert "hạn sử dụng" in cleaned


def test_entity_preservation_quoted_strings():
    # Chuỗi trong ngoặc kép không bao giờ bị biến đổi
    q = 'Cửa hàng bán được bao nhiêu "Trà sen vàng (M)" và "Cà phê sữa đá (L)" trong tháng 5?'
    cleaned = VietnameseNormalizer.clean_deterministic(q)
    assert '"Trà sen vàng (M)"' in cleaned
    assert '"Cà phê sữa đá (L)"' in cleaned


def test_entity_preservation_size_markers_and_codes():
    # Hậu tố size không ngoặc kép
    q = "Cho tôi số lượng ly bán ra của Trà đào cam sả (M) và Trà vải (L)"
    cleaned = VietnameseNormalizer.clean_deterministic(q)
    assert "(M)" in cleaned
    assert "(L)" in cleaned

    # Mã hóa đơn và mã sản phẩm
    q_code = "Kiểm tra tình trạng đơn HD202601 và sản phẩm SP987"
    cleaned_code = VietnameseNormalizer.clean_deterministic(q_code)
    assert "HD202601" in cleaned_code
    assert "SP987" in cleaned_code


def test_extract_cell_samples_dart_sql():
    df = pd.DataFrame({
        "Tên sản phẩm": ["Trà sữa trân châu hoàng gia", "Cà phê đen đá", "Sinh tố bơ"],
        "Khách hàng": ["Nguyễn Văn An", "Trần Thị Bích", "Lê Hoàng Long"],
    })
    tables = {"menu": df}

    samples = VietnameseNormalizer.extract_cell_samples("Khách An mua trà sữa bao nhiêu tiền?", tables=tables)
    assert "Trà sữa trân châu hoàng gia" in samples
    assert "Nguyễn Văn An" in samples


def test_additional_business_abbreviations():
    # Kiểm tra mở rộng từ viết tắt thương mại
    text = "Kiểm tra đvt, hsd, nsx của sp này và sl tồn kho cùng mức ck đc áp dụng"
    cleaned = VietnameseNormalizer.clean_deterministic(text)
    assert "đơn vị tính" in cleaned
    assert "hạn sử dụng" in cleaned
    assert "ngày sản xuất" in cleaned
    assert "sản phẩm" in cleaned
    assert "số lượng" in cleaned
    assert "chiết khấu" in cleaned
    assert "được" in cleaned


def test_entity_preservation_alphanumeric_codes_and_hyphens():
    # Đảm bảo mã hóa đơn có dấu gạch ngang, gạch dưới, hashtag không bị biến dạng
    text = 'Đơn hàng "HD-2026-01" và mã hàng "SP_009" giá bn?'
    cleaned = VietnameseNormalizer.clean_deterministic(text)
    assert '"HD-2026-01"' in cleaned
    assert '"SP_009"' in cleaned
    assert "bao nhiêu" in cleaned


def test_cell_samples_empty_and_no_match():
    # Trích xuất ô khi không có bảng hoặc câu hỏi không khớp bất kỳ ô nào
    assert VietnameseNormalizer.extract_cell_samples("Hôm nay ngày mấy?", tables={}) == ""
    assert VietnameseNormalizer.extract_cell_samples("Câu hỏi không có trong bảng", tables=None) == ""

    df = pd.DataFrame({"Số lượng": [1, 2, 3]})
    tables = {"kho": df}
    # Cột 'Số lượng' không phải cột văn bản tên/sản phẩm/khách -> trả về rỗng
    assert VietnameseNormalizer.extract_cell_samples("Trà sữa bao nhiêu?", tables=tables) == ""


def test_vietnamese_normalizer_strip_accents():
    from src.rule_engine import RuleEngine
    # Kiểm tra loại bỏ dấu tiếng Việt chính xác (đưa về chữ thường không dấu)
    raw = "Học sinh giỏi xuất sắc tiếng Việt: Đoàn Chuẩn, Nguyễn Ái Quốc"
    stripped = RuleEngine.strip_accents(raw)
    assert "hoc sinh gioi xuat sac tieng viet: doan chuan, nguyen ai quoc" == stripped
