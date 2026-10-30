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


DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parent.parent / "train" / "calendar.csv"


class DateResolver:
    """Bộ giải quyết các mốc thời gian theo đúng Mục 2 của Rulebook."""

    def __init__(self, calendar_path: Path = DEFAULT_CALENDAR_PATH) -> None:
        self.calendar_path = calendar_path
        self.lunar_to_solar: Dict[Tuple[int, int, int, bool], date] = {}
        self.solar_info: Dict[date, Dict[str, Any]] = {}
        self._load_calendar()

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
        start_dt = self.lunar_to_solar.get((tet_year - 1, 12, 23, False))
        if start_dt is None:
            start_dt = self.lunar_to_solar.get((tet_year - 1, 12, 23, True))
        end_dt = self.lunar_to_solar.get((tet_year, 1, 10, False))
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

        # Tuần bắt đầu thứ Hai (weekday() trong Python: Monday=0 .. Sunday=6)
        this_week_start = today - timedelta(days=today.weekday())
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

        # 0. Phát hiện phương ngữ Hôm nay / Hôm qua / Tuần này / Tuần trước
        if re.search(r"\b(hôm\s*nay|hom\s*nay|hnay|hôm\s*ni|hom\s*ni|bữa\s*nay|bua\s*nay|bữa\s*ni|bua\s*ni)\b", q_lower):
            detected.append(
                {
                    "expr": "Hôm nay (bao gồm phương ngữ hom ni / bữa nay / hnay)",
                    "start_date": today.isoformat(),
                    "end_date": today.isoformat(),
                    "note": f"Chính là ngày hôm nay ({today.isoformat()})",
                }
            )

        if re.search(r"\b(hôm\s*qua|hom\s*qua|bữa\s*qua|bua\s*qua)\b", q_lower):
            detected.append(
                {
                    "expr": "Hôm qua (bao gồm phương ngữ bữa qua / hom qua)",
                    "start_date": yesterday.isoformat(),
                    "end_date": yesterday.isoformat(),
                    "note": f"Chính là ngày hôm qua ({yesterday.isoformat()})",
                }
            )

        if re.search(r"\b(tuần\s*(?:trước|rồi|vừa\s*qua)|tuan\s*(?:truoc|roi|vua\s*qua))\b", q_lower):
            detected.append(
                {
                    "expr": "Tuần trước / tuần rồi",
                    "start_date": last_week_start.isoformat(),
                    "end_date": last_week_end.isoformat(),
                    "note": f"Cả tuần liền trước từ Thứ Hai ({last_week_start.isoformat()}) đến Chủ Nhật ({last_week_end.isoformat()})",
                }
            )

        if re.search(r"\b(tuần\s*này|tuan\s*nay)\b", q_lower):
            detected.append(
                {
                    "expr": "Tuần này",
                    "start_date": this_week_start.isoformat(),
                    "end_date": this_week_end.isoformat(),
                    "note": f"Từ Thứ Hai ({this_week_start.isoformat()}) đến hết hôm nay ({this_week_end.isoformat()})",
                }
            )

        # 1. Phát hiện N ngày gần nhất / N ngày qua
        for m in re.finditer(r"(\d+)\s*ngày\s*(?:gần\s*nhất|gần\s*đây|qua|vừa\s*qua)", q_lower):
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

        # 3. Phát hiện ngày cụ thể DD/MM/YYYY hoặc DD-MM-YYYY
        for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", q_lower):
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
        for m in re.finditer(r"(?<!quý\s)(?<!quy\s)(?<!quý)(?<!quy)\b(?:tháng|thang|th|t)?\s*(\d{1,2})[/-](20\d{2}|\d{2})\b", q_lower):
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

        # 5. Phát hiện tháng M không ghi năm ("tháng M" không theo sau bởi /YYYY hoặc /YY)
        for m in re.finditer(r"\b(?:tháng|thang|th|t)\s*(\d{1,2})(?![/\-\d])\b", q_lower):
            m_val = int(m.group(1))
            if 1 <= m_val <= 12:
                # Mục 2 Rulebook: của năm nay nếu không sau hôm nay; sau hôm nay thì của năm trước
                candidate_year = today.year if m_val <= today.month else today.year - 1
                s_dt, e_dt = self.get_month_range(candidate_year, m_val)
                if candidate_year == today.year and m_val == today.month:
                    e_dt = today
                detected.append(
                    {
                        "expr": m.group(0),
                        "start_date": s_dt.isoformat(),
                        "end_date": e_dt.isoformat(),
                        "year_month": f"{candidate_year:04d}-{m_val:02d}",
                        "note": f"Tháng {m_val} không ghi năm -> {m_val:02d}/{candidate_year}",
                    }
                )

        # 6. Phát hiện Quý Q/YYYY hoặc Quý Q không ghi năm
        for m in re.finditer(r"(?:quý|quy)\s*([1-4])(?:[/-](20\d{2})|\s+năm\s+(20\d{2}))?", q_lower):
            q_val = int(m.group(1))
            y_str = m.group(2) or m.group(3)
            if y_str:
                q_year = int(y_str)
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

        # 7. Phát hiện Tết (Tết năm Y, dịp Tết, mùa Tết)
        if re.search(r"\b(tết|tet)\b", q_lower):
            tet_year_match = re.search(r"(?:tết|tet)\s*(?:năm|nam|âm\s*lịch)?\s*(20\d{2})", q_lower)
            if tet_year_match:
                tet_years = [int(tet_year_match.group(1))]
            else:
                # Nếu không ghi năm, cung cấp cả Tết năm nay và năm trước theo as_of_date
                tet_years = sorted({today.year - 1, today.year})
            for ty in tet_years:
                tet_rng = self.get_tet_range(ty)
                pre_rng = self.get_pre_tet_range(ty)
                if tet_rng:
                    detected.append(
                        {
                            "expr": f"Tết {ty}",
                            "start_date": tet_rng[0].isoformat(),
                            "end_date": tet_rng[1].isoformat(),
                            "note": f"Tết âm lịch {ty} (23 tháng Chạp {ty-1} đến mùng 10 tháng Giêng {ty})",
                        }
                    )
                if pre_rng:
                    detected.append(
                        {
                            "expr": f"Trước Tết {ty}",
                            "start_date": pre_rng[0].isoformat(),
                            "end_date": pre_rng[1].isoformat(),
                            "note": f"Trước Tết {ty} (cùng số ngày ngay trước 23 tháng Chạp)",
                        }
                    )

        result["detected_periods"] = detected
        return result

    def format_for_prompt(self, question: str, as_of_date_str: str) -> str:
        """Định dạng thông tin thời gian thành đoạn văn bản rõ ràng để đưa vào Prompt LLM."""
        ctx = self.resolve(question, as_of_date_str)
        lines = [
            f"- Hôm nay (`as_of_date`): {ctx['as_of_date']} (Mọi bản ghi có ngày > {ctx['as_of_date']} coi như chưa xảy ra, trừ trạng thái huỷ/trả ghi trên đơn).",
            f"- Hôm qua: {ctx['yesterday'][0]}",
            f"- Tuần này (Thứ Hai -> Hôm nay): {ctx['this_week'][0]} đến {ctx['this_week'][1]}",
            f"- Tuần trước / tuần rồi (Thứ Hai -> Chủ Nhật): {ctx['last_week'][0]} đến {ctx['last_week'][1]}",
            f"- Tháng này (đầu tháng -> Hôm nay): {ctx['this_month'][0]} đến {ctx['this_month'][1]}",
            f"- Tháng trước (cả tháng): {ctx['prev_month_full'][0]} đến {ctx['prev_month_full'][1]}",
            f"- Kỳ 'so với tháng trước' (khi so với 'tháng này' - cùng số ngày): {ctx['prev_month_same_days'][0]} đến {ctx['prev_month_same_days'][1]}",
            f"- Từ đầu năm tới nay: {ctx['ytd'][0]} đến {ctx['ytd'][1]}",
        ]
        if ctx["detected_periods"]:
            lines.append("- Các mốc thời gian phát hiện trực tiếp trong câu hỏi:")
            for p in ctx["detected_periods"]:
                lines.append(f"  * {p['expr']}: từ {p['start_date']} đến {p['end_date']} ({p['note']})")
        return "\n".join(lines)
