# -*- coding: utf-8 -*-
"""
Unit tests cho Module 2A: Date Resolver & Calendar Lookup.
Kiểm tra giải quyết mốc thời gian, ngày đầu tuần, tháng trước, và mùa Tết âm lịch.
"""

from datetime import date
from pathlib import Path
import pytest
from src.date_resolver import DateResolver


CALENDAR_PATH = Path(__file__).resolve().parent.parent / "assets" / "calendar.csv"


def test_date_resolver_relative_today_and_yesterday():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    ctx = resolver.resolve("Hôm nay bán được bao nhiêu tiền?", as_of_date_str="2026-05-15")
    assert ctx["as_of_date"] == "2026-05-15"
    assert ctx["today"] == ("2026-05-15", "2026-05-15")
    assert ctx["yesterday"] == ("2026-05-14", "2026-05-14")


def test_date_resolver_week_starts_monday():
    # 2026-05-15 là thứ Sáu (Friday)
    # Thứ Hai đầu tuần là 2026-05-11
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    ctx = resolver.resolve("Tuần này doanh thu bao nhiêu?", as_of_date_str="2026-05-15")
    assert ctx["this_week"] == ("2026-05-11", "2026-05-15")
    # Tuần trước: cả tuần liền trước (thứ Hai 04/05 đến Chủ nhật 10/05)
    assert ctx["last_week"] == ("2026-05-04", "2026-05-10")


def test_date_resolver_this_month_and_prev_month():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    ctx = resolver.resolve("Tháng này so với tháng trước tăng bao nhiêu?", as_of_date_str="2026-05-15")
    # Tháng này: từ ngày 1 đến hết hôm nay
    assert ctx["this_month"] == ("2026-05-01", "2026-05-15")
    # Tháng trước cả tháng: 01/04 đến 30/04
    assert ctx["prev_month_full"] == ("2026-04-01", "2026-04-30")
    # Tháng trước cùng số ngày: 01/04 đến 15/04
    assert ctx["prev_month_same_days"] == ("2026-04-01", "2026-04-15")


def test_date_resolver_tet_range_from_calendar():
    if not CALENDAR_PATH.exists():
        pytest.skip("calendar.csv không tồn tại trong assets")
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    # Tết năm 2026: 23 tháng Chạp năm Ất Tỵ đến mùng 10 tháng Giêng năm Bính Ngọ
    tet_range = resolver.get_tet_range(2026)
    assert tet_range is not None
    start_tet, end_tet = tet_range
    assert start_tet < end_tet
    # Kiểm tra trước Tết
    pre_range = resolver.get_pre_tet_range(2026)
    assert pre_range is not None
    pre_start, pre_end = pre_range
    assert (end_tet - start_tet).days == (pre_end - pre_start).days


def test_date_resolver_quarter_ranges():
    q1_s, q1_e = DateResolver.get_quarter_range(2025, 1)
    assert q1_s == date(2025, 1, 1) and q1_e == date(2025, 3, 31)

    q2_s, q2_e = DateResolver.get_quarter_range(2025, 2)
    assert q2_s == date(2025, 4, 1) and q2_e == date(2025, 6, 30)

    q3_s, q3_e = DateResolver.get_quarter_range(2025, 3)
    assert q3_s == date(2025, 7, 1) and q3_e == date(2025, 9, 30)

    q4_s, q4_e = DateResolver.get_quarter_range(2025, 4)
    assert q4_s == date(2025, 10, 1) and q4_e == date(2025, 12, 31)


def test_date_resolver_leap_year_february():
    # 2024 là năm nhuận (tháng 2 có 29 ngày)
    m24_s, m24_e = DateResolver.get_month_range(2024, 2)
    assert m24_s == date(2024, 2, 1)
    assert m24_e == date(2024, 2, 29)

    # 2025 không phải năm nhuận (tháng 2 có 28 ngày)
    m25_s, m25_e = DateResolver.get_month_range(2025, 2)
    assert m25_s == date(2025, 2, 1)
    assert m25_e == date(2025, 2, 28)


def test_date_resolver_dialects_and_aliases():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    # Phương ngữ hom ni / bữa nay
    ctx_hn = resolver.resolve("Bữa nay thu được bao nhiêu?", as_of_date_str="2026-05-15")
    assert any("2026-05-15" in p.get("start_date", "") for p in ctx_hn["detected_periods"])

    # Phương ngữ bữa qua / hôm qua
    ctx_hq = resolver.resolve("Bữa qua bán mấy ly trà?", as_of_date_str="2026-05-15")
    assert any("2026-05-14" in p.get("start_date", "") for p in ctx_hq["detected_periods"])

    # Hôm kia (2 ngày trước: 2026-05-13)
    ctx_hk = resolver.resolve("Hôm kia có khách nào nợ tiền không?", as_of_date_str="2026-05-15")
    assert any("2026-05-13" in p.get("start_date", "") for p in ctx_hk["detected_periods"])


def test_date_resolver_two_weeks_ago():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    # as_of_date = 2026-05-15 (thứ Sáu tuần 2026-05-11 đến 2026-05-15)
    # tuần trước: 2026-05-04 đến 2026-05-10
    # hai tuần trước: 2026-04-27 đến 2026-05-03
    ctx = resolver.resolve("Doanh thu hai tuần trước thế nào?", as_of_date_str="2026-05-15")
    assert ctx["two_weeks_ago"] == ("2026-04-27", "2026-05-03")
    assert ctx["hai tuần trước"] == ("2026-04-27", "2026-05-03")


def test_date_resolver_january_boundary_rollback():
    resolver = DateResolver(calendar_path=CALENDAR_PATH)
    # as_of_date rơi vào đầu tháng 1 (ví dụ 2026-01-08)
    ctx = resolver.resolve("So với tháng trước thì thế nào?", as_of_date_str="2026-01-08")
    # Tháng trước của tháng 1 năm 2026 phải là tháng 12 năm 2025
    assert ctx["prev_month_full"] == ("2025-12-01", "2025-12-31")
    assert ctx["prev_month_same_days"] == ("2025-12-01", "2025-12-08")
    assert ctx["ytd"] == ("2026-01-01", "2026-01-08")
