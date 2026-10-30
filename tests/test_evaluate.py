# -*- coding: utf-8 -*-
"""Unit tests kiểm chứng bộ chấm điểm evaluate.py theo Mục 7 & Mục 9 Rulebook."""

from pathlib import Path
from evaluate import (
    evaluate_predictions,
    load_jsonl,
    score_single_prediction,
)


TRAIN_ANSWERS = Path(__file__).resolve().parent.parent / "train" / "answers.jsonl"


def test_self_evaluation_on_3000_train_answers():
    """Chấm chính tập answers.jsonl chuẩn phải đạt 3000/3000 (PA = 100%)."""
    gold = load_jsonl(TRAIN_ANSWERS)
    assert len(gold) == 3000
    report = evaluate_predictions(gold, gold)
    assert report["total"] == 3000
    assert report["correct"] == 3000
    assert report["wrong"] == 0
    assert report["abstain"] == 0
    assert report["penalized_accuracy"] == 1.0


def test_abstain_and_missing():
    gold = {"id": "q1", "type": "number", "answer": 1000, "alias": []}
    assert score_single_prediction(None, gold) == (0, "abstain")
    assert score_single_prediction({"id": "q1", "type": "abstain"}, gold) == (0, "abstain")


def test_number_and_percent_strict_format():
    gold_num = {"id": "q1", "type": "number", "answer": 4820000, "alias": []}
    assert score_single_prediction({"id": "q1", "type": "number", "answer": 4820000}, gold_num) == (1, "correct")
    assert score_single_prediction({"id": "q1", "type": "number", "answer": "4820000"}, gold_num) == (-1, "wrong")
    assert score_single_prediction({"id": "q1", "type": "number", "answer": True}, gold_num) == (-1, "wrong")
    assert score_single_prediction({"id": "q1", "type": "percent", "answer": 4820000}, gold_num) == (-1, "wrong")


def test_string_and_alias_matching():
    gold_str = {
        "id": "q_str",
        "type": "string",
        "answer": "Costar Omega 3 1000mg lọ 100 viên",
        "alias": ["SP0076", "Costar Omega 3 1000 mg lọ 100 viên"],
    }
    assert score_single_prediction(
        {"id": "q_str", "type": "string", "answer": "costar omega 3 1000 mg, lọ 100 viên!"},
        gold_str,
    ) == (1, "correct")
    assert score_single_prediction(
        {"id": "q_str", "type": "string", "answer": "sp0076"},
        gold_str,
    ) == (1, "correct")
    assert score_single_prediction(
        {"id": "q_str", "type": "string", "answer": "Costar Omega 3 500mg"},
        gold_str,
    ) == (-1, "wrong")


def test_list_ordering_and_element_aliases():
    gold_list = {
        "id": "q_list",
        "type": "list",
        "answer": ["Cà phê sữa (M)", "Trà đào cam sả (M)"],
        "alias": [["SP0003"], ["SP0017"]],
    }
    assert score_single_prediction(
        {"id": "q_list", "type": "list", "answer": ["SP0003", "Trà đào cam sả (m)"]},
        gold_list,
    ) == (1, "correct")
    # Sai thứ tự -> -1
    assert score_single_prediction(
        {"id": "q_list", "type": "list", "answer": ["Trà đào cam sả (M)", "Cà phê sữa (M)"]},
        gold_list,
    ) == (-1, "wrong")


def test_table_row_order_independent_and_aliases():
    gold_table = {
        "id": "q_tbl",
        "type": "table",
        "answer": {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158600]],
        },
        "alias": [["KH0124"], ["KH0233"]],
    }
    pred_table = {
        "id": "q_tbl",
        "type": "table",
        "answer": {
            "columns": ["ten_khach", "no"],
            "rows": [["KH0233", 158600], ["đỗ đức an", 404500]],
        },
    }
    assert score_single_prediction(pred_table, gold_table) == (1, "correct")


def test_unanswerable_reason_rule():
    gold_unans = {"id": "q_u", "type": "unanswerable", "answer": None, "alias": []}
    assert score_single_prediction(
        {"id": "q_u", "type": "unanswerable", "reason": "Ngoài khoảng thời gian dữ liệu"},
        gold_unans,
    ) == (1, "correct")
    # Mục 7 Rulebook: reason không phải chuỗi -> không đọc được và tính như bỏ trống (0 điểm)
    assert score_single_prediction(
        {"id": "q_u", "type": "unanswerable", "reason": 12345},
        gold_unans,
    ) == (0, "abstain")
