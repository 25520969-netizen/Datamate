# -*- coding: utf-8 -*-
"""
Unit tests cho evaluate.py: Bộ chấm điểm chuẩn Penalized Accuracy.
Kiểm tra tính điểm Đúng (+1), Sai (-1), Bỏ qua/Abstain (0) và các hàm so sánh giá trị.
"""

import pytest
from evaluate import (
    evaluate_predictions,
    compare_numbers,
    compare_strings,
    compare_dates,
    compare_lists,
    is_json_number,
)


def test_is_json_number():
    assert is_json_number(123)
    assert is_json_number(12.34)
    assert not is_json_number(True)  # bool không phải json number
    assert not is_json_number(False)
    assert not is_json_number("123")
    assert not is_json_number(float("nan"))
    assert not is_json_number(float("inf"))


def test_compare_numbers():
    assert compare_numbers(100, 100)
    assert compare_numbers(100.000001, 100.0, tol=1e-5)
    assert not compare_numbers(100, 101)


def test_compare_strings():
    assert compare_strings("Sting dâu 330ml", "sting dâu 330ml")
    assert compare_strings("Cà phê sữa đá", "CA PHE SUA DA") is False  # Dấu tiếng Việt phải đúng theo Mục 9
    assert compare_strings("Sting dâu 330ml", "Sting dâu 330 ml")
    assert compare_strings("SP01", "Nước ngọt", aliases=["SP01", "Sting"])


def test_compare_dates():
    assert compare_dates("2026-05-15", "2026-05-15")
    assert compare_dates("2026-05", "2026-05")
    assert not compare_dates("15/05/2026", "2026-05-15")  # sai format


def test_compare_lists():
    assert compare_lists(["A", "B"], ["A", "B"])
    assert compare_lists([10, 20], [10, 20])
    assert not compare_lists(["B", "A"], ["A", "B"])  # sai thứ tự
    assert not compare_lists([10], [10, 20])  # khác độ dài


def test_evaluate_predictions_penalized_accuracy():
    preds = [
        {"id": "q1", "type": "number", "answer": 100},      # Đúng -> +1
        {"id": "q2", "type": "number", "answer": 999},      # Sai -> -1
        {"id": "q3", "type": "abstain"},                    # Abstain -> 0
        {"id": "q4", "type": "string", "answer": "Món A"},   # Đúng -> +1
        # q5 vắng mặt -> 0
    ]
    gold = [
        {"id": "q1", "type": "number", "answer": 100},
        {"id": "q2", "type": "number", "answer": 200},
        {"id": "q3", "type": "number", "answer": 300},
        {"id": "q4", "type": "string", "answer": "Món A"},
        {"id": "q5", "type": "number", "answer": 500},
    ]

    report = evaluate_predictions(preds, gold)
    assert report["total"] == 5
    assert report["correct"] == 2
    assert report["wrong"] == 1
    assert report["abstain"] == 2
    # Raw points = 2 - 1 = 1
    assert report["raw_points"] == 1
    # Penalized Accuracy = 1 / 5 = 0.20
    assert report["penalized_accuracy"] == 0.20
