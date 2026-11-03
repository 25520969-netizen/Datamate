# -*- coding: utf-8 -*-
"""
Module 2A: Date Resolver (Bộ giải quyết thời gian theo Mục 2 Rulebook & calendar.csv).
Quy đổi chính xác các cách diễn đạt thời gian tương đối/tuyệt đối trong câu hỏi tiếng Việt
dựa trên `as_of_date` ("hôm nay") và bảng lịch âm-dương `calendar.csv`.
"""

import calendar
import csv
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parent.parent / "assets" / "calendar.csv"


class DateResolver:
    """Bộ giải quyết các mốc thời gian theo đúng Mục 2 của Rulebook."""

    def __init__(self, calendar_path: Path = DEFAULT_CALENDAR_PATH) -> None:
        self.calendar_path = calendar_path
        self.lunar_to_solar: Dict[Tuple[int, int, int, bool], date] = {}
        self.solar_info: Dict[date, Dict[str, Any]] = {}
        # Tham số lấy từ rulebook đang dùng (apply_knobs); giá trị khởi tạo chỉ là mặc định khi rulebook không ghi
        self.week_start = 0
        self.tet_start = (12, 23)
        self.tet_end = (1, 10)
        self._load_calendar()

    WEEKDAY_NAMES = {"thứ hai": 0, "thứ ba": 1, "thứ tư": 2, "thứ năm": 3, "thứ sáu": 4, "thứ bảy": 5, "chủ nhật": 6}

    def apply_knobs(self, knobs: Any) -> None:
        """Đọc ngày đầu tuần và khoảng Tết từ rulebook (RulebookKnobs) thay vì gắn cứng."""
        if knobs is None:
            return
        wk = str(getattr(knobs, "week_start_day", "")).strip().lower()
        if wk in self.WEEKDAY_NAMES:
            self.week_start = self.WEEKDAY_NAMES[wk]
        self.tet_start = (int(getattr(knobs, "tet_start_lunar_month", 12)), int(getattr(knobs, "tet_start_lunar_day", 23)))
        self.tet_end = (int(getattr(knobs, "tet_end_lunar_month", 1)), int(getattr(knobs, "tet_end_lunar_day", 10)))

    def _load_calendar(self) -> None:
        if not self.calendar_path.exists():
            return
        with self.calendar_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dt = date.fromisoformat(row["date"].strip())
                    l_day = int(row["lunar_day"])
                    l_month = int(row["lunar_month"])
                    l_year = int(row["lunar_year"])
                    l_leap = row.get("lunar_leap", "false").strip().lower() == "true"
                    self.lunar_to_solar[(l_year, l_month, l_day, l_leap)] = dt
                    self.solar_info[dt] = {
                        "weekday": int(row["weekday"]),
                        "lunar_day": l_day,
                        "lunar_month": l_month,
                        "lunar_year": l_year,
                        "lunar_leap": l_leap,
                        "holiday": row.get("holiday", "").strip(),
                    }
                except (KeyError, ValueError):
                    continue

    @staticmethod
    def parse_iso_date(date_str: str) -> date:
        return date.fromisoformat(date_str.strip()[:10])

    @staticmethod
    def get_month_range(year: int, month: int) -> Tuple[date, date]:
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, 1), date(year, month, last_day)

    @staticmethod
    def get_quarter_range(year: int, quarter: int) -> Tuple[date, date]:
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        start_dt = date(year, start_month, 1)
        end_dt = date(year, end_month, calendar.monthrange(year, end_month)[1])
        return start_dt, end_dt

    def get_tet_range(self, tet_year: int) -> Optional[Tuple[date, date]]:
        """
        Mục 2 Rulebook:
        - Tết năm Y, dịp Tết, mùa Tết: từ ngày dương của 23 tháng Chạp (năm âm Y - 1)
          đến hết ngày dương của mùng 10 tháng Giêng (năm âm Y).
        """
        s_month, s_day = self.tet_start
        e_month, e_day = self.tet_end
        s_year = tet_year - 1 if s_month > e_month else tet_year
        start_dt = None
        for d in range(s_day, max(s_day - 2, 0), -1):
            # Tháng âm thiếu (29 ngày) không có ngày 30 -> lấy ngày cuối tháng
            start_dt = self.lunar_to_solar.get((s_year, s_month, d, False)) or self.lunar_to_solar.get((s_year, s_month, d, True))
            if start_dt:
                break
        end_dt = self.lunar_to_solar.get((tet_year, e_month, e_day, False))
        if start_dt and end_dt:
            return start_dt, end_dt
        return None

    def get_pre_tet_range(self, tet_year: int) -> Optional[Tuple[date, date]]:
        """
        Mục 2 Rulebook:
        - So với trước Tết = cùng số ngày ngay trước 23 tháng Chạp.
        """
        tet_range = self.get_tet_range(tet_year)
        if not tet_range:
            return None
        tet_start, tet_end = tet_range
        num_days = (tet_end - tet_start).days + 1
        pre_end = tet_start - timedelta(days=1)
        pre_start = pre_end - timedelta(days=num_days - 1)
        return pre_start, pre_end

    def resolve(self, question: str, as_of_date_str: str) -> Dict[str, Any]:
        """
        Trích xuất và tính toán sẵn tất cả các mốc thời gian chuẩn từ `as_of_date`
        và từ nội dung câu hỏi để cung cấp cho LLM và Sandbox.
        """
        today = self.parse_iso_date(as_of_date_str)
        yesterday = today - timedelta(days=1)

        # Ngày đầu tuần theo rulebook (weekday() trong Python: Monday=0 .. Sunday=6)
        this_week_start = today - timedelta(days=(today.weekday() - self.week_start) % 7)
        this_week_end = today
        last_week_end = this_week_start - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)

        # Tháng này: từ ngày 1 đến hết hôm nay
        this_month_start = date(today.year, today.month, 1)
        this_month_end = today

        # Tháng trước (cả tháng) và "so với tháng trước" (cùng số ngày với tháng này)
        if today.month == 1:
            prev_m_year, prev_m_month = today.year - 1, 12
        else:
            prev_m_year, prev_m_month = today.year, today.month - 1

        prev_month_full_start, prev_month_full_end = self.get_month_range(prev_m_year, prev_m_month)
        prev_month_max_day = calendar.monthrange(prev_m_year, prev_m_month)[1]
        prev_month_same_days_end = date(prev_m_year, prev_m_month, min(today.day, prev_month_max_day))

        # Từ đầu năm
        ytd_start = date(today.year, 1, 1)
        ytd_end = today

        result: Dict[str, Any] = {
            "as_of_date": today.isoformat(),
            "today": (today.isoformat(), today.isoformat()),
            "yesterday": (yesterday.isoformat(), yesterday.isoformat()),
            "this_week": (this_week_start.isoformat(), this_week_end.isoformat()),
            "last_week": (last_week_start.isoformat(), last_week_end.isoformat()),
            "two_weeks_ago": ((last_week_start - timedelta(days=7)).isoformat(), (last_week_end - timedelta(days=7)).isoformat()),
            "two_weeks_before": ((last_week_start - timedelta(days=7)).isoformat(), (last_week_end - timedelta(days=7)).isoformat()),
            "hai tuần trước": ((last_week_start - timedelta(days=7)).isoformat(), (last_week_end - timedelta(days=7)).isoformat()),
            "this_month": (this_month_start.isoformat(), this_month_end.isoformat()),
            "prev_month_full": (prev_month_full_start.isoformat(), prev_month_full_end.isoformat()),
            "prev_month_same_days": (prev_month_full_start.isoformat(), prev_month_same_days_end.isoformat()),
            "ytd": (ytd_start.isoformat(), ytd_end.isoformat()),
            # Các khóa tiếng Việt tương đương để tránh KeyError khi LLM truy cập trực tiếp
            "hôm nay": (today.isoformat(), today.isoformat()),
            "hôm qua": (yesterday.isoformat(), yesterday.isoformat()),
            "tuần này": (this_week_start.isoformat(), this_week_end.isoformat()),
            "tuần trước": (last_week_start.isoformat(), last_week_end.isoformat()),
            "tuần rồi": (last_week_start.isoformat(), last_week_end.isoformat()),
            "tháng này": (this_month_start.isoformat(), this_month_end.isoformat()),
            "tháng trước": (prev_month_full_start.isoformat(), prev_month_full_end.isoformat()),
            "từ đầu năm": (ytd_start.isoformat(), ytd_end.isoformat()),
            # Các khóa chuỗi đơn (start / end) thường được LLM truy cập trực tiếp
            "month_start": this_month_start.isoformat(),
            "month_end": this_month_end.isoformat(),
            "this_month_start": this_month_start.isoformat(),
            "this_month_end": this_month_end.isoformat(),
            "week_start": this_week_start.isoformat(),
            "week_end": this_week_end.isoformat(),
            "this_week_start": this_week_start.isoformat(),
            "this_week_end": this_week_end.isoformat(),
            "last_week_start": last_week_start.isoformat(),
            "last_week_end": last_week_end.isoformat(),
            "prev_month_start": prev_month_full_start.isoformat(),
            "prev_month_end": prev_month_full_end.isoformat(),
            "year_start": ytd_start.isoformat(),
            "year_end": ytd_end.isoformat(),
            "ytd_start": ytd_start.isoformat(),
            "ytd_end": ytd_end.isoformat(),
            "detected_periods": [],
        }

        q_lower = question.lower()
        detected: List[Dict[str, str]] = []

        # 0. Phát hiện phương ngữ Hôm nay / Hôm qua / Hôm kia / Tuần này / Tuần trước / Tháng này / Tháng trước / Năm nay / Năm ngoái
        if re.search(r"\b(hôm\s*nay|hom\s*nay|hnay|hôm\s*ni|hom\s*ni|bữa\s*nay|bua\s*nay|bữa\s*ni|bua\s*ni)\b", q_lower):
            detected.append(
                {
                    "expr": "Hôm nay (bao gồm phương ngữ hom ni / bữa nay / hnay)",
                    "start_date": today.isoformat(),
                    "end_date": today.isoformat(),
                    "note": f"Chính là ngày hôm nay ({today.isoformat()})",
                }
            )

        if re.search(r"\b(hôm\s*qua|hom\s*qua|bữa\s*qua|bua\s*qua|hqua|h\/qua|ngay\s*qua)\b", q_lower):
            detected.append(
                {
                    "expr": "Hôm qua (bao gồm phương ngữ bữa qua / hom qua / hqua)",
                    "start_date": yesterday.isoformat(),
                    "end_date": yesterday.isoformat(),
                    "note": f"Chính là ngày hôm qua ({yesterday.isoformat()})",
                }
            )

        if re.search(r"\b(hôm\s*kia|hom\s*kia)\b", q_lower):
            day_before_yesterday = today - timedelta(days=2)
            detected.append(
                {
                    "expr": "Hôm kia (2 ngày trước)",
                    "start_date": day_before_yesterday.isoformat(),
                    "end_date": day_before_yesterday.isoformat(),
                    "note": f"Ngày hôm kia ({day_before_yesterday.isoformat()})",
                }
            )

        if re.search(r"\b((?:2|hai)\s*tuần\s*(?:trước|trc|rồi|vừa\s*qua)|(?:2|hai)\s*tuan\s*(?:truoc|trc|roi|vua\s*qua)|tuần\s*(?:trước|trc)\s*nữa|tuan\s*(?:truoc|trc)\s*nua|tuần\s*kia)\b", q_lower):
            two_w_start = last_week_start - timedelta(days=7)
            two_w_end = last_week_end - timedelta(days=7)
            detected.append(
                {
                    "expr": "Hai tuần trước (2 tuần trước)",
                    "start_date": two_w_start.isoformat(),
                    "end_date": two_w_end.isoformat(),
                    "note": f"Cả tuần cách đây 2 tuần ({two_w_start.isoformat()} đến {two_w_end.isoformat()})",
                }
            )
        elif re.search(r"\b(tuần\s*(?:trước|trc|rồi|vừa\s*qua)|tuan\s*(?:truoc|trc|roi|vua\s*qua))\b", q_lower):
            detected.append(
                {
                    "expr": "Tuần trước / tuần rồi",
                    "start_date": last_week_start.isoformat(),
                    "end_date": last_week_end.isoformat(),
                    "note": f"Cả tuần liền trước ({last_week_start.isoformat()} đến {last_week_end.isoformat()})",
                }
            )

        if re.search(r"\b(tuần\s*(?:này|nay)|tuan\s*(?:nay))\b", q_lower):
            detected.append(
                {
                    "expr": "Tuần này",
                    "start_date": this_week_start.isoformat(),
                    "end_date": this_week_end.isoformat(),
                    "note": f"Từ đầu tuần ({this_week_start.isoformat()}) đến hết hôm nay ({this_week_end.isoformat()})",
                }
            )

        if re.search(r"\b(tháng\s*(?:này|nay|hiện\s*tại)|thang\s*(?:nay|hien\s*tai)|thg\s*nay|th\s*nay)\b", q_lower):
            detected.append(
                {
                    "expr": "Tháng này (từ đầu tháng đến hôm nay)",
                    "start_date": this_month_start.isoformat(),
                    "end_date": this_month_end.isoformat(),
                    "note": f"Từ đầu tháng ({this_month_start.isoformat()}) đến hết hôm nay ({this_month_end.isoformat()})",
                }
            )

        if re.search(r"\b(tháng\s*(?:trước|trc|rồi)|thang\s*(?:truoc|trc|roi)|thg\s*trc|th\s*trc)\b", q_lower):
            if not re.search(r"\b(so\s*với\s*tháng|so\s*vs\s*tháng|so\s*tháng|so\s*vs\s*th)\b", q_lower):
                detected.append(
                    {
                        "expr": "Tháng trước (cả tháng)",
                        "start_date": prev_month_full_start.isoformat(),
                        "end_date": prev_month_full_end.isoformat(),
                        "note": f"Toàn bộ tháng liền trước ({prev_month_full_start.isoformat()} đến {prev_month_full_end.isoformat()})",
                    }
                )
            else:
                detected.append(
                    {
                        "expr": "Kỳ so với tháng trước (cùng số ngày với tháng này)",
                        "start_date": prev_month_full_start.isoformat(),
                        "end_date": prev_month_same_days_end.isoformat(),
                        "note": f"Từ ngày 01 đến ngày cùng số của tháng trước ({prev_month_full_start.isoformat()} đến {prev_month_same_days_end.isoformat()})",
                    }
                )

        # "Cùng kỳ năm ngoái/năm trước" là kỳ hỏi lùi 1 năm (xử lý ở cuối hàm), không phải cả năm ngoái
        is_same_period_last_year = bool(re.search(
            r"\b(?:cùng\s*kỳ|cung\s*ky)\s*(?:của\s*|cua\s*)?(?:năm|nam)\s*(?:ngoái|ngoai|trước|truoc|trc)\b", q_lower
        ))

        # "tháng 3 năm ngoái", "quý 2 năm nay": năm tương đối bổ nghĩa cho tháng/quý (xử lý ở mục 5/6), không phải cả năm
        rel_year_of_month = re.search(
            r"\b(?:tháng|thang|thg|th|quý|quy)\s*\d{1,2}\s*(?:của\s*|cua\s*)?(?:năm|nam)\s*(ngoái|ngoai|trước|truoc|trc|nay)\b", q_lower
        )
        rel_year_offset = None
        if rel_year_of_month:
            rel_year_offset = 0 if rel_year_of_month.group(1) == "nay" else -1

        if is_same_period_last_year or rel_year_of_month:
            pass
        elif re.search(r"\b(?:từ\s*đầu\s*năm|tu\s*dau\s*nam|đầu\s*năm\s*(?:đến|tới)\s*nay)\b", q_lower):
            detected.append(
                {
                    "expr": "Từ đầu năm tới nay",
                    "start_date": ytd_start.isoformat(),
                    "end_date": today.isoformat(),
                    "note": f"Từ 01/01 ({ytd_start.isoformat()}) đến hết hôm nay ({today.isoformat()})",
                }
            )
        elif re.search(r"\b(năm\s*nay|nam\s*nay)\b", q_lower):
            detected.append(
                {
                    "expr": "Từ đầu năm nay tới nay",
                    "start_date": ytd_start.isoformat(),
                    "end_date": today.isoformat(),
                    "note": f"Từ đầu năm ({ytd_start.isoformat()}) đến hết hôm nay ({today.isoformat()})",
                }
            )
        elif re.search(r"\b(năm\s*ngoái|nam\s*ngoai|năm\s*(?:trước|trc)|nam\s*(?:truoc|trc))\b", q_lower):
            prev_y = today.year - 1
            detected.append(
                {
                    "expr": f"Cả năm ngoái ({prev_y})",
                    "start_date": f"{prev_y}-01-01",
                    "end_date": f"{prev_y}-12-31",
                    "note": f"Toàn bộ năm {prev_y}",
                }
            )

        # 1. Phát hiện N ngày gần nhất / N ngày qua (kể cả không dấu hoặc gõ Telex: 7 ngay gan nhat, 30 ngay qua)
        for m in re.finditer(r"(\d+)\s*(?:ngày|ngay)\s*(?:gần\s*nhất|gan\s*nhat|gần\s*đây|gan\s*day|qua|vừa\s*qua|vua\s*qua)", q_lower):
            n_days = int(m.group(1))
            if n_days > 0:
                s_dt = today - timedelta(days=n_days - 1)
                detected.append(
                    {
                        "expr": m.group(0),
                        "start_date": s_dt.isoformat(),
                        "end_date": today.isoformat(),
                        "note": f"{n_days} ngày liền nhau kể cả hôm nay ({today.isoformat()})",
                    }
                )

        # 2. Phát hiện N ngày tới (ví dụ: hết hạn trong N ngày tới: từ D đến hết D + N)
        for m in re.finditer(r"(\d+)\s*ngày\s*(?:tới|tiếp\s*theo|sắp\s*tới)", q_lower):
            n_days = int(m.group(1))
            e_dt = today + timedelta(days=n_days)
            detected.append(
                {
                    "expr": m.group(0),
                    "start_date": today.isoformat(),
                    "end_date": e_dt.isoformat(),
                    "note": f"Từ hôm nay D ({today.isoformat()}) đến hết ngày D + {n_days} ({e_dt.isoformat()})",
                }
            )

        # 2b. Phát hiện dải ngày cụ thể: Từ DD/MM/YYYY đến (hết) DD/MM/YYYY
        range_match = re.search(r"\btừ\s*(?:ngày\s*)?(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\s*(?:đến|tới|den|toi)\s*(?:hết\s*)?(?:ngày\s*)?(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", q_lower)
        if range_match:
            try:
                s_dt = date(int(range_match.group(3)), int(range_match.group(2)), int(range_match.group(1)))
                e_dt = date(int(range_match.group(6)), int(range_match.group(5)), int(range_match.group(4)))
                detected.append(
                    {
                        "expr": range_match.group(0),
                        "start_date": s_dt.isoformat(),
                        "end_date": e_dt.isoformat(),
                        "note": f"Dải ngày từ {s_dt.isoformat()} đến hết {e_dt.isoformat()}",
                    }
                )
            except ValueError:
                pass

        # 3. Phát hiện ngày cụ thể DD/MM/YYYY hoặc DD-MM-YYYY (nếu chưa nằm trong dải ngày trên)
        for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", q_lower):
            if range_match and m.start() >= range_match.start() and m.end() <= range_match.end():
                continue
            d_val, m_val, y_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                exact_dt = date(y_val, m_val, d_val)
                detected.append(
                    {
                        "expr": m.group(0),
                        "start_date": exact_dt.isoformat(),
                        "end_date": exact_dt.isoformat(),
                        "note": f"Ngày cụ thể {exact_dt.isoformat()}",
                    }
                )
            except ValueError:
                pass

        # 4. Phát hiện tháng M/YYYY hoặc M/YY (bao gồm thM/YYYY, tM/YY, tháng M/YYYY, ví dụ: th6/2025, 3/25, t5/2026; loại trừ quý M/YYYY)
        for m in re.finditer(r"(?<!quý\s)(?<!quy\s)(?<!quý)(?<!quy)(?<![\d/\-])\b(?:tháng|thang|thg|th|t)?\s*(?<![\d/\-])(\d{1,2})[/-](20\d{2}|\d{2})\b(?![/\-]\d)", q_lower):
            m_val = int(m.group(1))
            y_val = int(m.group(2))
            if y_val < 100:
                y_val += 2000
            if 1 <= m_val <= 12 and 2020 <= y_val <= 2035:
                s_dt, e_dt = self.get_month_range(y_val, m_val)
                detected.append(
                    {
                        "expr": m.group(0).strip(),
                        "start_date": s_dt.isoformat(),
                        "end_date": e_dt.isoformat(),
                        "year_month": f"{y_val:04d}-{m_val:02d}",
                        "note": f"Cả tháng {m_val:02d}/{y_val}",
                    }
                )

        # 4b. Phát hiện tháng M năm YYYY (ví dụ: tháng 5 năm 2026, thang 10 nam 2025)
        for m in re.finditer(r"(?<!quý\s)(?<!quy\s)(?<!quý)(?<!quy)\b(?:tháng|thang|thg|th|t)\s*(\d{1,2})\s*(?:năm|nam)\s*(20\d{2})\b", q_lower):
            m_val = int(m.group(1))
            y_val = int(m.group(2))
            if 1 <= m_val <= 12 and 2020 <= y_val <= 2035:
                s_dt, e_dt = self.get_month_range(y_val, m_val)
                ym_str = f"{y_val:04d}-{m_val:02d}"
                if not any(d.get("year_month") == ym_str for d in detected):
                    detected.append(
                        {
                            "expr": m.group(0).strip(),
                            "start_date": s_dt.isoformat(),
                            "end_date": e_dt.isoformat(),
                            "year_month": ym_str,
                            "note": f"Cả tháng {m_val:02d}/{y_val}",
                        }
                    )

        # 5. Phát hiện tháng M không ghi năm ("tháng M" không theo sau bởi /YYYY, /YY hoặc năm YYYY)
        for m in re.finditer(r"\b(?:tháng|thang|thg|th|t)\s*(\d{1,2})(?![/\-\d])(?!\s*(?:năm|nam)\s*\d{4})\b", q_lower):
            m_val = int(m.group(1))
            if 1 <= m_val <= 12:
                # Mục 2 Rulebook: của năm nay nếu không sau hôm nay; sau hôm nay thì của năm trước
                candidate_year = today.year if m_val <= today.month else today.year - 1
                if rel_year_offset is not None:
                    candidate_year = today.year + rel_year_offset
                s_dt, e_dt = self.get_month_range(candidate_year, m_val)
                if candidate_year == today.year and m_val == today.month:
                    e_dt = today
                ym_str = f"{candidate_year:04d}-{m_val:02d}"
                if not any(d.get("year_month") == ym_str for d in detected):
                    detected.append(
                        {
                            "expr": m.group(0),
                            "start_date": s_dt.isoformat(),
                            "end_date": e_dt.isoformat(),
                            "year_month": ym_str,
                            "note": f"Tháng {m_val} không ghi năm -> {m_val:02d}/{candidate_year}",
                        }
                    )

        # 6. Phát hiện Quý Q/YYYY hoặc Quý Q không ghi năm
        for m in re.finditer(r"(?:quý|quy)\s*([1-4])(?:[/-](20\d{2})|\s+năm\s+(20\d{2}))?", q_lower):
            q_val = int(m.group(1))
            y_str = m.group(2) or m.group(3)
            if y_str:
                q_year = int(y_str)
            elif rel_year_offset is not None:
                q_year = today.year + rel_year_offset
            else:
                q_start_month = (q_val - 1) * 3 + 1
                q_year = today.year if date(today.year, q_start_month, 1) <= today else today.year - 1
            s_dt, e_dt = self.get_quarter_range(q_year, q_val)
            if e_dt > today and not y_str:
                e_dt = today
            detected.append(
                {
                    "expr": m.group(0),
                    "start_date": s_dt.isoformat(),
                    "end_date": e_dt.isoformat(),
                    "note": f"Quý {q_val}/{q_year}",
                }
            )

        # 6b. Quý này (từ đầu quý đến hôm nay) / quý trước (cả quý liền trước)
        cur_q = (today.month - 1) // 3 + 1
        if re.search(r"\b(?:quý|quy)\s*(?:này|nay)\b", q_lower):
            qs, _ = self.get_quarter_range(today.year, cur_q)
            detected.append({"expr": "Quý này", "start_date": qs.isoformat(), "end_date": today.isoformat(),
                             "note": f"Từ đầu quý {cur_q}/{today.year} đến hết hôm nay"})
        if re.search(r"\b(?:quý|quy)\s*(?:trước|truoc|trc|rồi|roi)\b", q_lower):
            pq, py = (cur_q - 1, today.year) if cur_q > 1 else (4, today.year - 1)
            qs, qe = self.get_quarter_range(py, pq)
            detected.append({"expr": "Quý trước", "start_date": qs.isoformat(), "end_date": qe.isoformat(),
                             "note": f"Cả quý {pq}/{py}"})

        # 6c. Năm YYYY đứng riêng (không đi sau tháng/quý/Tết): cả năm, năm nay thì đến hết hôm nay
        for m in re.finditer(r"(?:năm|nam)\s*(20\d{2})\b", q_lower):
            prefix = q_lower[max(0, m.start() - 12):m.start()]
            if re.search(r"(?:tháng|thang|thg|quý|quy|tết|tet)\s*\d*\s*$", prefix):
                continue
            y_val = int(m.group(1))
            e_dt = min(date(y_val, 12, 31), today) if y_val == today.year else date(y_val, 12, 31)
            detected.append({"expr": m.group(0), "start_date": f"{y_val}-01-01", "end_date": e_dt.isoformat(),
                             "note": f"Cả năm {y_val}"})

        # 6d. Ngày D/M không ghi năm ("ngày 5/6", "từ 01/05 đến 15/05"): năm nay nếu không sau hôm nay, ngược lại năm trước
        dm_matches = list(re.finditer(r"(?<![\d/\-])(\d{1,2})[/-](\d{1,2})(?![/\-]?\d)", q_lower))
        dm_dates = []
        for m in dm_matches:
            d_val, m_val = int(m.group(1)), int(m.group(2))
            if not (1 <= m_val <= 12 and 1 <= d_val <= 31) or "ngày" not in q_lower and "ngay" not in q_lower and "từ" not in q_lower:
                continue
            try:
                cand = date(today.year, m_val, d_val)
                if cand > today:
                    cand = date(today.year - 1, m_val, d_val)
            except ValueError:
                continue
            dm_dates.append(cand)
        if len(dm_dates) == 2 and re.search(r"\btừ\b.*\b(?:đến|tới)\b", q_lower) and dm_dates[0] <= dm_dates[1]:
            detected.append({"expr": "Từ ngày đến ngày (không ghi năm)", "start_date": dm_dates[0].isoformat(),
                             "end_date": dm_dates[1].isoformat(), "note": "Gồm cả hai đầu mút"})
        else:
            for cand in dm_dates:
                detected.append({"expr": f"ngày {cand.day}/{cand.month}", "start_date": cand.isoformat(),
                                 "end_date": cand.isoformat(), "note": f"Ngày {cand.isoformat()} (không ghi năm)"})

        # 7. Phát hiện Tết (Tết năm Y, dịp Tết, mùa Tết)
        if re.search(r"\b(tết|tet)\b", q_lower):
            tet_year_match = re.search(r"(?:tết|tet)\s*(?:năm|nam|âm\s*lịch)?\s*(20\d{2})", q_lower)
            if tet_year_match:
                tet_years = [int(tet_year_match.group(1))]
            else:
                # Không ghi năm: lấy dịp Tết gần nhất đã bắt đầu tính đến hôm nay (Tết chưa tới thì chưa có dữ liệu)
                started = [
                    ty for ty in (today.year + 1, today.year, today.year - 1)
                    if self.get_tet_range(ty) and self.get_tet_range(ty)[0] <= today
                ]
                tet_years = [started[0]] if started else [today.year]
            for ty in tet_years:
                tet_rng = self.get_tet_range(ty)
                pre_rng = self.get_pre_tet_range(ty)
                if tet_rng:
                    detected.append(
                        {
                            "expr": f"Tết {ty}",
                            "start_date": tet_rng[0].isoformat(),
                            "end_date": tet_rng[1].isoformat(),
                            "note": f"Tết âm lịch {ty} theo khoảng Tết trong rulebook",
                        }
                    )
                # Kỳ "trước Tết" chỉ thêm khi câu hỏi có so sánh, tránh kỳ thừa làm lệch kiểm tra "ngoài vùng dữ liệu"
                if pre_rng and re.search(r"trước\s*tết|truoc\s*tet|so\s*(?:với|voi|vs)", q_lower):
                    detected.append(
                        {
                            "expr": f"Trước Tết {ty}",
                            "start_date": pre_rng[0].isoformat(),
                            "end_date": pre_rng[1].isoformat(),
                            "note": f"Trước Tết {ty} (cùng số ngày ngay trước ngày bắt đầu Tết)",
                        }
                    )

        if is_same_period_last_year:
            base_periods = [p for p in detected if p.get("start_date") and p.get("end_date")]
            if not base_periods:
                # "cùng kỳ năm ngoái" không kèm kỳ nào khác -> kỳ mặc định là từ đầu năm đến hôm nay
                base_periods = [{"expr": "Từ đầu năm tới nay", "start_date": ytd_start.isoformat(), "end_date": today.isoformat()}]
                detected.extend(dict(p, note=f"Kỳ hỏi ({p['start_date']} đến {p['end_date']})") for p in base_periods)
            for p in base_periods:
                s_prev = self._shift_one_year_back(date.fromisoformat(p["start_date"]))
                e_prev = self._shift_one_year_back(date.fromisoformat(p["end_date"]))
                detected.append(
                    {
                        "expr": f"Cùng kỳ năm trước của '{p['expr']}'",
                        "start_date": s_prev.isoformat(),
                        "end_date": e_prev.isoformat(),
                        "is_comparison_baseline": True,
                        "note": f"Kỳ so sánh: lùi đúng 1 năm ({s_prev.isoformat()} đến {e_prev.isoformat()})",
                    }
                )

        result["detected_periods"] = detected
        return result

    @staticmethod
    def _shift_one_year_back(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            # 29/02 -> 28/02 năm trước
            return d.replace(year=d.year - 1, day=28)

    def format_for_prompt(self, question: str, as_of_date_str: str) -> str:
        """Định dạng thông tin thời gian thành đoạn văn bản rõ ràng để đưa vào Prompt LLM."""
        ctx = self.resolve(question, as_of_date_str)
        lines = []
        if ctx["detected_periods"]:
            lines.append("Mốc thời gian quy đổi sẵn từ câu hỏi (tham khảo; nếu khác mục Thời gian của rulebook thì theo rulebook):")
            for p in ctx["detected_periods"]:
                lines.append(f"  * {p['expr']}: từ '{p['start_date']}' đến '{p['end_date']}' ({p['note']})")
            lines.append("")

        lines.append(f"- Hôm nay (`as_of_date`): {ctx['as_of_date']}.")

        # Khử nhiễu: Nếu câu hỏi đã xác định được mốc mục tiêu cụ thể và không phải là câu hỏi so sánh tương đối giữa 2 kỳ,
        # ẩn bớt các mốc tham chiếu khác (tuần này, tháng trước, YTD...) để tránh LLM bị loãng chú ý và lọc nhầm khoảng thời gian.
        q_low = question.lower()
        is_comparative = any(k in q_low for k in ["so với", "so vs", "cùng kỳ", "cung ky", "hơn hay kém", "hon hay kem", "tăng hay giảm", "tang hay giam", "chênh", "chenh"])
        if not ctx["detected_periods"] or is_comparative:
            lines.extend([
                f"- Hôm qua: {ctx['yesterday'][0]}",
                f"- Tuần này (đầu tuần -> hôm nay): {ctx['this_week'][0]} đến {ctx['this_week'][1]}",
                f"- Tuần trước / tuần rồi (cả tuần): {ctx['last_week'][0]} đến {ctx['last_week'][1]}",
                f"- Tháng này (đầu tháng -> Hôm nay): {ctx['this_month'][0]} đến {ctx['this_month'][1]}",
                f"- Tháng trước (cả tháng): {ctx['prev_month_full'][0]} đến {ctx['prev_month_full'][1]}",
                f"- Kỳ 'so với tháng trước' (khi so với 'tháng này' - cùng số ngày): {ctx['prev_month_same_days'][0]} đến {ctx['prev_month_same_days'][1]}",
                f"- Từ đầu năm tới nay: {ctx['ytd'][0]} đến {ctx['ytd'][1]}",
            ])
        return "\n".join(lines)
