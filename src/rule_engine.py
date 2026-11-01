# -*- coding: utf-8 -*-
"""
Module 2B: Query Analyzer & Rule Knowledge Injector.
- Phân loại kiểu đáp án kỳ vọng (number, percent, date, boolean, string, list, table).
- Trích xuất đúng các mục luật liên quan từ `train/rulebook.md` và `README.md` của shop.
- Kiểm tra nhanh dấu hiệu câu hỏi không trả lời được (`unanswerable`) khi kỳ hỏi nằm hoàn toàn ngoài vùng dữ liệu.
"""

import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và đưa về chữ thường để so khớp thực thể mềm."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).lower().strip()


class DynamicEntityGrounder:
    """
    Dynamic Entity Grounding Engine (theo kiến trúc CHESS VLDB 2024 / BIRD NeurIPS 2023).
    Tự động xây dựng từ điển thực thể (sản phẩm, dịch vụ, khách hàng, kênh bán, đơn vị tính)
    từ toàn bộ bảng dữ liệu của shop trong RAM.
    Khi người dùng hỏi bằng tiếng Việt không dấu hoặc viết tắt, engine tìm kiếm thực thể gốc
    và đơn vị tính tương ứng để gợi ý chính xác cho LLM trong Prompt.
    Hoàn toàn dynamic, không hardcode bất kỳ danh mục hay sản phẩm của shop cụ thể nào.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}

    def _build_index_for_business(self, business_id: str, tables: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, str]]]:
        if business_id in self._cache:
            return self._cache[business_id]

        entity_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for tname, df in tables.items():
            if not hasattr(df, "columns") or len(df) == 0:
                continue

            # Tìm các cột định danh thực thể (sản phẩm, dịch vụ, khách hàng, nhân viên, kênh)
            name_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in [
                    "tên", "ten", "hàng", "hang", "sản phẩm", "san_pham",
                    "mặt hàng", "mat_hang", "món", "mon", "khách", "khach",
                    "nhân viên", "nhan_vien", "kênh", "kenh", "chi nhánh", "chi_nhanh"
                ])
                and not any(ex in c.lower() for ex in ["mã", "id", "ngày", "ngay", "tiền", "tien", "giá", "gia", "thuế", "chiết khấu"])
            ]
            # Tìm cột đơn vị tính nếu có
            dvt_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in ["đvt", "dvt", "đơn vị", "don_vi", "quy cách", "unit"])
            ]
            dvt_col = dvt_cols[0] if dvt_cols else None

            for nc in name_cols:
                subset_cols = [nc] + ([dvt_col] if dvt_col else [])
                for _, row in df[subset_cols].drop_duplicates().iterrows():
                    orig = str(row[nc]).strip()
                    if not orig or orig.lower() in ["none", "nan", "", "<na>", "null"]:
                        continue
                    dvt = str(row[dvt_col]).strip() if dvt_col and pd.notna(row[dvt_col]) else ""
                    key = strip_accents(orig)
                    # Chỉ lập chỉ mục các chuỗi từ 3 ký tự trở lên để tránh nhiễu
                    if len(key) >= 3:
                        if key not in entity_map:
                            entity_map[key] = []
                        entry = (orig, dvt, nc)
                        if entry not in entity_map[key]:
                            entity_map[key].append(entry)

        self._cache[business_id] = entity_map
        return entity_map

    def find_grounding_hints(self, question: str, business_id: str, tables: Dict[str, Any]) -> str:
        """
        Tìm kiếm các thực thể trong câu hỏi và sinh đoạn gợi ý schema grounding cho Prompt.
        """
        entity_map = self._build_index_for_business(business_id, tables)
        if not entity_map:
            return ""

        q_clean = strip_accents(question)
        matched_hints: List[str] = []
        matched_keys: set = set()

        sorted_keys = sorted(entity_map.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if re.search(r"(?:^|[^\w])" + re.escape(key) + r"(?:[^\w]|$)", q_clean):
                if any(key in mk for mk in matched_keys):
                    continue
                matched_keys.add(key)
                candidates = entity_map[key]

                best_cand = candidates[0]
                if len(candidates) > 1:
                    for cand in candidates:
                        if cand[1] and strip_accents(cand[1]) in q_clean:
                            best_cand = cand
                            break

                orig_name, dvt, col_name = best_cand
                dvt_str = f", ĐVT: '{dvt}'" if dvt else ""
                matched_hints.append(f"- '{orig_name}' (cột '{col_name}'{dvt_str})")
                if len(matched_hints) >= 5:
                    break

        # Vòng 2: Khớp cụm từ con (subphrase n-gram >= 2 từ, >= 8 ký tự)
        # Cho phép các câu hỏi viết rút gọn ("nep cai hoa vang", "lau ga la e") khớp đúng thực thể trong DB
        if len(matched_hints) < 5:
            q_tokens = {t for t in re.split(r"[\s_]+", q_clean) if len(t) >= 2}
            for key in sorted_keys:
                if len(matched_hints) >= 5:
                    break
                if any(key in mk for mk in matched_keys):
                    continue
                clean_key = re.sub(r"\(.*?\)", "", key).strip()
                toks = [t for t in re.split(r"[\s_]+", clean_key) if t]
                if len(toks) < 2:
                    continue
                # Tối ưu: Nếu không có từ nào của key xuất hiện trong câu hỏi thì bỏ qua ngay lập tức
                if not any(t in q_tokens for t in toks if len(t) >= 3):
                    continue
                matched = False
                for n in range(len(toks), 1, -1):
                    for i in range(len(toks) - n + 1):
                        subphrase = " ".join(toks[i : i + n])
                        if len(subphrase) >= 8 and re.search(r"(?:^|[^\w])" + re.escape(subphrase) + r"(?:[^\w]|$)", q_clean):
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    matched_keys.add(key)
                    candidates = entity_map[key]
                    best_cand = candidates[0]
                    orig_name, dvt, col_name = best_cand
                    dvt_str = f", ĐVT: '{dvt}'" if dvt else ""
                    matched_hints.append(f"- '{orig_name}' (cột '{col_name}'{dvt_str})")

        if not matched_hints:
            return ""

        return "=== GỢI Ý THỰC THỂ KHỚP TRONG CƠ SỞ DỮ LIỆU ===\n" + "\n".join(matched_hints)


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "train" / "rulebook.md"


class RulebookKnobs:
    """Các tham số cấu hình (núm) đọc động từ rulebook.md của split tương ứng."""
    def __init__(self) -> None:
        self.week_start_day: str = "thứ Hai"
        self.shift_morning: Tuple[str, str] = ("06:00", "13:59")
        self.shift_afternoon: Tuple[str, str] = ("14:00", "21:59")
        self.tet_start_lunar_day: int = 23
        self.tet_start_lunar_month: int = 12
        self.tet_end_lunar_day: int = 10
        self.tet_end_lunar_month: int = 1
        self.money_decimals: int = 0
        self.percent_decimals: int = 2
        self.excluded_statuses: List[str] = ["Đã hủy", "Trả hàng"]

    def __repr__(self) -> str:
        return (
            f"RulebookKnobs(week_start='{self.week_start_day}', "
            f"morning={self.shift_morning}, afternoon={self.shift_afternoon}, "
            f"money_dec={self.money_decimals}, pct_dec={self.percent_decimals})"
        )


class RulebookKnobReader:
    """Trích xuất tự động các tham số cấu hình từ file rulebook.md mà không hardcode."""

    @staticmethod
    def parse_knobs(rulebook_text: str) -> RulebookKnobs:
        knobs = RulebookKnobs()
        if not rulebook_text:
            return knobs

        # 1. Ngày đầu tuần
        m_week = re.search(r"Tuần.*?bắt đầu\s+\*\*([^\*]+)\*\*", rulebook_text, re.IGNORECASE)
        if m_week:
            knobs.week_start_day = m_week.group(1).strip()

        # 2. Khung giờ ca sáng / ca chiều
        m_morning = re.search(r"ca sáng\s+(\d{1,2}:\d{2})[–-](\d{1,2}:\d{2})", rulebook_text, re.IGNORECASE)
        if m_morning:
            knobs.shift_morning = (m_morning.group(1), m_morning.group(2))

        m_afternoon = re.search(r"ca chiều\s+(\d{1,2}:\d{2})[–-](\d{1,2}:\d{2})", rulebook_text, re.IGNORECASE)
        if m_afternoon:
            knobs.shift_afternoon = (m_afternoon.group(1), m_afternoon.group(2))

        # 3. Mùa Tết âm lịch
        m_tet_start = re.search(r"(\d{1,2})\s+tháng Chạp", rulebook_text, re.IGNORECASE)
        if m_tet_start:
            knobs.tet_start_lunar_day = int(m_tet_start.group(1))

        m_tet_end = re.search(r"mùng\s+(\d{1,2})\s+tháng Giêng", rulebook_text, re.IGNORECASE)
        if m_tet_end:
            knobs.tet_end_lunar_day = int(m_tet_end.group(1))

        # 4. Làm tròn tiền và phần trăm
        if "làm tròn về đồng" in rulebook_text.lower():
            knobs.money_decimals = 0
        m_pct_dec = re.search(r"phần trăm.*?làm tròn\s+(\d+)\s+chữ số thập phân", rulebook_text, re.IGNORECASE)
        if m_pct_dec:
            knobs.percent_decimals = int(m_pct_dec.group(1))

        return knobs


class RuleEngine:
    """Bộ phân tích câu hỏi và trích xuất luật nghiệp vụ theo BIRD Knowledge Injection paradigm."""

    def __init__(self, rulebook_path: Path = DEFAULT_RULEBOOK_PATH) -> None:
        self.rulebook_path = rulebook_path
        self.sections: Dict[str, str] = {}
        self.full_rulebook: str = ""
        self.knobs: RulebookKnobs = RulebookKnobs()
        self._load_rulebook()

    def _load_rulebook(self) -> None:
        if not self.rulebook_path.exists():
            return
        self.full_rulebook = self.rulebook_path.read_text(encoding="utf-8")
        self.knobs = RulebookKnobReader.parse_knobs(self.full_rulebook)

        current_header = "intro"
        buf: List[str] = []
        for line in self.full_rulebook.splitlines():
            if line.startswith("## "):
                self.sections[current_header] = "\n".join(buf).strip()
                current_header = line.lstrip("#").strip()
                buf = [line]
            elif line.startswith("### ") and current_header.startswith("6. Theo ngành"):
                self.sections[current_header] = "\n".join(buf).strip()
                sub_title = line.lstrip("#").strip()
                current_header = f"6. Theo ngành - {sub_title}"
                buf = [line]
            else:
                buf.append(line)
        self.sections[current_header] = "\n".join(buf).strip()

    @staticmethod
    def normalize_telex_typos(s: str) -> str:
        """Chuẩn hóa một số lỗi gõ Telex phổ biến: naof -> nào, thangs -> tháng, cuwar -> cửa, nhaf -> nhà, v.v."""
        s = re.sub(r"\bnaof\b", "nào", s)
        s = re.sub(r"\bthangs\b", "tháng", s)
        s = re.sub(r"\bcuwar\b", "cửa", s)
        s = re.sub(r"\bnhaf\b", "nhà", s)
        s = re.sub(r"\bduocj\b", "được", s)
        s = re.sub(r"\bbani\b", "bán", s)
        s = re.sub(r"\btieenf\b", "tiền", s)
        s = re.sub(r"\bcuoois\b", "cuối", s)
        s = re.sub(r"\bcuar\b", "của", s)
        s = re.sub(r"\bnhoms\b", "nhóm", s)
        s = re.sub(r"\blaf\b", "là", s)
        s = re.sub(r"\bhown\b", "hơn", s)
        s = re.sub(r"\btrar\b", "trả", s)
        s = re.sub(r"\bhees\b", "hết", s)
        s = re.sub(r"\bngayf\b", "ngày", s)
        s = re.sub(r"\bnhaats\b", "nhất", s)
        s = re.sub(r"\btrawm\b", "trăm", s)
        s = re.sub(r"\bnhieeuf\b", "nhiều", s)
        s = re.sub(r"\bnhieeu\b", "nhiều", s)
        s = re.sub(r"\bbieets\b", "biết", s)
        s = re.sub(r"\bvaayj\b", "vậy", s)
        return s

    @staticmethod
    def infer_expected_type(question: str) -> str:
        """
        Dự đoán kiểu đáp án kỳ vọng từ câu hỏi tiếng Việt (number, percent, date, boolean, string, list, table).
        Lưu ý: Đây là gợi ý cho LLM; kiểu cuối cùng có thể được code Python tự xác định hoặc chuẩn hóa.
        """
        q = RuleEngine.normalize_telex_typos(question.lower().strip())

        # 1. Table: bảng / mỗi ... kèm số tiền / danh sách ... và doanh thu/số tiền/số lượng/công nợ
        if (
            re.search(
                r"\b(mỗi người còn nợ bao nhiêu|mỗi người nợ bao nhiêu|mỗi người nợ bn|mỗi khách nợ bao nhiêu|"
                r"mỗi người bao nhiêu tiền nợ|số nợ của từng người|mỗi người đang nợ bao nhiêu|"
                r"moi nguoi con no|moi nguoi no|tung nguoi con no|moi nguoi bnhiu|"
                r"kèm\s*(?:theo\s*)?(?:số\s*tiền\s*)?(?:còn\s*)?nợ|kèm\s*dư\s*nợ|"
                r"kem\s*(?:theo\s*)?(?:so\s*tien\s*)?(?:con\s*)?no|kem\s*du\s*no|"
                r"và\s*(?:số\s*tiền\s*)?(?:còn\s*)?nợ|va\s*(?:so\s*tien\s*)?(?:con\s*)?no)\b",
                q,
            )
            or re.search(r"\b(?:top\s*\d+|danh\s*sách|bảng)\b.*\b(?:kèm|và)\s+(?:số\s+tiền|doanh\s+thu|số\s+lượng|số\s+nợ|tiền\s+nợ|công\s+nợ|hoa\s+hồng)", q)
            or (
                re.search(r"\b(?:top\s*\d+|[2-9]|10)\s+(?:khách|khach)\b", q)
                and any(k in q for k in ["nợ nhiều nhất", "nợ nhìu nhất", "công nợ lớn nhất", "no lon nhat", "no nhieu nhat"])
                and any(k in q for k in ["mỗi người", "từng người", "mỗi khách", "và số nợ", "số tiền nợ", "moi nguoi", "tung nguoi", "kèm", "kem"])
            )
            or any(
                kw in q
                for kw in [
                    "bảng ",
                    "lập bảng",
                    "kèm số tiền",
                    "kèm doanh thu",
                    "kèm số nợ",
                    "kèm công nợ",
                    "kèm số lượng",
                    "mỗi khách còn nợ bao nhiêu",
                    "từng khách còn nợ bao nhiêu",
                    "từng người còn nợ",
                    "từng người nha",
                    "mỗi người còn nợ",
                    "số nợ của từng",
                    "nợ của từng người",
                    "nợ của mỗi người",
                ]
            )
        ):
            return "table"

        # 2. Boolean: có phải ... không, đúng hay sai, tăng liên tục, luôn cao hơn
        # Loại trừ 'hok kê đơn' / 'không kê đơn' tránh nhận nhầm câu hỏi thuốc là boolean
        q_bool = re.sub(r"\b(?:hàng|thuốc|hang|thuoc)?\s*hok\s*(?:kê\s*đơn|ke\s*don)\b", "", q)
        if (
            re.search(
                r"\b(có phải|co phai|có đúng|co dung|đúng không|dung khong|phải không|phai khong|không bạn|khong ban|"
                r"không nhỉ|khong nhi|không shop|khong shop|hay không|hay khong|liệu có|lieu co|hok|k hả|k ha|"
                r"đúng hay sai|có hay không|co hay khong)\b",
                q_bool,
            )
            or q_bool.endswith(" không?") or q_bool.endswith(" khong?") or q_bool.endswith(" k?") or q_bool.endswith(" ko?")
            or re.search(r"\b(?:có\s+)?(?:tăng\s+liên\s+tục|tang\s+lien\s+tuc|luôn\s+cao\s+hơn|luon\s+cao\s+hon)\s+.*(?:không|khong|k|ko|hok)", q_bool)
        ):
            if not re.search(r"\b(bao nhiêu|nhiêu|mấy|may|ai|gì|gi|nào|nao)\b", q_bool) or re.search(r"\b(?:tăng\s+liên\s+tục|tang\s+lien\s+tuc|luôn\s+cao\s+hơn)\b", q_bool):
                return "boolean"

        # 3. Percent: phần trăm, tỉ lệ, tỷ lệ, tỉ trọng, tỷ trọng, bao nhiêu %, mấy %, tăng/giảm bao nhiêu %
        # Loại trừ % nồng độ sản phẩm trong câu hỏi (ví dụ: cồn 70%, nước muối 0,9%, otrivin 0,1%, kem 5%)
        q_no_pct = re.sub(r"\b\d+(?:[\.,]\d+)?\s*%", "", q)
        q_no_prod_pct = re.sub(r"(?:cồn|con|ruou|rượu|ethanol)\s*\d+[\.,]?\d*\s*%", "", q_no_pct)
        if (
            any(
                kw in q
                for kw in [
                    "phần trăm",
                    "phan tram",
                    "tỉ lệ",
                    "tỷ lệ",
                    "ti le",
                    "ty le",
                    "tỉ trọng",
                    "tỷ trọng",
                    "ti trong",
                    "ty trong",
                    "tỷ số",
                    "tỉ số",
                    "tỉ lệ đơn huỷ",
                    "tỉ lệ đơn hủy",
                    "tỷ lệ đơn huỷ",
                    "tỷ lệ đơn hủy",
                    "tỉ lệ huỷ",
                    "tỉ lệ hủy",
                    "tỉ lệ hoàn",
                    "tăng hay giảm bao nhiêu",
                    "tang hay giam bao nhieu",
                    "tăng hay giảm mấy",
                    "tang hay giam may",
                    "thay đổi mấy",
                    "thay doi may",
                    "thay đổi bao nhiêu",
                ]
            )
            or "%" in q_no_prod_pct
            or re.search(r"\b(?:tăng|giảm|thay đổi|hơn|kém|chiếm|đóng góp|đạt)\s+.*(?:nhiêu|bao nhiêu|bn|mấy)\s*%", q)
        ):
            return "percent"

        # 4. Date: ngày nào, tháng nào, hôm nào, bao giờ, khi nào, hạn dùng ngày nào
        if re.search(
            r"\b(ngày nào|ngay nao|ngày mấy|ngay may|hôm nào|hom nao|khi nào|khi nao|bao giờ|bao gio|"
            r"hạn dùng ngày nào|han dung ngay nao|hạn sử dụng là ngày|ngày bao nhiêu|ngay bao nhieu|"
            r"tháng nào|thang nao|vào tháng nào|vao thang nao|thang naof)\b",
            q,
        ):
            return "date"

        # 5. List:
        # a. Hỏi 2 hoặc nhiều giá trị: "mỗi loại bao nhiêu", "lần lượt là bao nhiêu", "theo thứ tự đó", "mấy ... chẵn ... lẻ", "tuổi nợ"
        # hoặc nhiều vế câu hỏi bao nhiêu, hoặc 2 doanh thu nối bằng "và" / "va"
        if (
            re.search(
                r"\b(mỗi loại bao nhiêu|mỗi loại bn|mỗi bên bao nhiêu|mỗi bên bn|mỗi thứ bao nhiêu|mỗi thứ bn|"
                r"moi loai bnhiu|moi loai bn|moi loai bao nhieu|moi ben bn|moi ben bao nhieu|"
                r"mỗi tháng là bao nhiêu|mỗi tháng là bnhiu|mỗi tháng bnhiu|moi thang la bao nhieu|moi thang la bnhiu|"
                r"lần lượt là bao nhiêu|lần lượt là bn|lần lượt theo thứ tự|theo thứ tự đó|theo thứ tự thời gian|theo thu tu thoi gian|"
                r"ghi (?:hai|ba|[2-9]) số|ghi (?:hai|ba|[2-9]) so|hai số|2 số|ba số|3 số|trả lời hai số|tra loi hai so|trả lời 2 số|tra loi 2 so|"
                r"ghi lần lượt|ghi lan luot|"
                r"chẵn.*lẻ|chan.*le|hộp chẵn|vỉ chẵn|thùng chẵn|"
                r"tuổi nợ|tuoi no|"
                r"từ tháng cũ đến tháng mới|tu thang cu den thang moi)\b",
                q,
            )
            or len(re.findall(r"\b(?:bao\s*(?:nhiêu|nhiu)|bn|bnhiu)\b", q)) >= 2
            or re.search(r"(?:dthu|doanh thu|tiền|tien).*?\b(?:và|va)\b.*?(?:dthu|doanh thu|tiền|tien)", q)
            or re.search(r"(?:dthu|doanh thu|tiền|tien).*?[,;]\s*(?:doanh thu|dthu|tiền|tien)", q)
            or re.search(r"(?:ghi\s*)?nợ\s*thêm.*?trả\s*(?:nợ\s*)?(?:được|dc|duoc)", q)
            or re.search(r"(?:no them|ghi no them).*?(?:tra dc|tra duoc)", q)
        ):
            return "list"

        # b. Hỏi danh sách đối tượng: những ai, top N, các mặt hàng nào, liệt kê, N khách mua nhiều nhất, cho tôi N ...
        if (
            re.search(
                r"\b(những .* nào|nhung .* nao|các .* nào|cac .* nao|liệt kê|liet ke|top\s*\d+|"
                r"(?:cho\s*(?:tôi|toi|mình|minh|em|xin)\s+)?(?<!\d)(?:[2-9]|10)\s+(?:mặt hàng|mat hang|khách hàng|khach hang|khách|khach|món|mon|sản phẩm|san pham|thợ|tho|nhân viên|nhan vien|thuốc|thuoc)\b|"
                r"bao nhiêu mỗi tháng|mỗi tháng là bao nhiêu|mấy thùng mấy|bao nhiêu thùng bao nhiêu|"
                r"(?:hàng|thuốc|mặt hàng|sản phẩm)\s+nào\s+hết\s+hạn)\b",
                q,
            )
            or bool(re.search(r"\b(?<!\d)(?:[2-9]|10)\s+(?:khách|khach|mặt hàng|mat hang|món|mon|sản phẩm|san pham|thợ|tho|nhân viên|nhan vien|thuốc|thuoc)\b", q))
        ):
            # Lưu ý: nếu hỏi số lượng mặt hàng ế ("mấy mặt hàng ế") thì là NUMBER
            if not re.search(r"\b(?:mấy|bao nhiêu)\s+(?:mặt hàng|sản phẩm|món|loại)\s+(?:ế|ees|bán chậm|không ai mua)\b", q):
                return "list"

        # Đếm số lượng mặt hàng ế / thuốc / khách ("mấy mặt hàng ế", "có bao nhiêu mặt hàng không bán được") -> Luôn là NUMBER
        if re.search(r"\b(?:mấy|bao nhiêu)\s+(?:mặt hàng|sản phẩm|món|loại)\s+(?:ế|ees|bán chậm|không ai mua)\b", q):
            return "number"

        # 6. String: hỏi đích danh 1 đối tượng duy nhất (ai, hàng nào, sản phẩm nào, món nào, khách nào, ncc nào, là gì, là chi)
        if re.search(
            r"\b(ai là|ai la|ai đang|ai dang|ai trung thành|ai sộp|ai hay|ai được|ai bán|ai mua|ai có|ai chi|là ai|la ai|"
            r"mặt hàng nào|mat hang nao|hàng nào|hang nao|sản phẩm nào|san pham nao|món nào|mon nao|"
            r"khách hàng nào|khach hang nao|khách nào|khach nao|nhà cung cấp nào|nha cung cap nao|ncc nào|ncc nao|nhân viên nào|thợ nào|tho nao|"
            r"lô nào|lo nao|nhóm nào|kênh nào|hình thức nào|chi nhánh nào|thứ mấy|thu may|hàng hot nhất|mặt hàng hot nhất|"
            r"là gì|la gi|là chi|la chi)\b",
            q,
        ):
            # Tránh nhầm "bán được mấy ... là gì"
            if not re.search(r"\b(bao nhiêu (?:tiền|đồng|cái|chai|lon|kg|hộp)|nhiêu tiền)\b", q):
                return "string"

        return "number"

    @staticmethod
    def detect_unanswerable_deterministic(
        question_obj: Dict[str, Any],
        store: Any,
        date_resolver: Any,
    ) -> Tuple[bool, str]:
        """
        Kiểm tra tiền định theo Mục 7 Rulebook:
        - Kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop.
        - Shop không ghi loại thông tin câu hỏi cần (địa chỉ, giới tính, tồn kho, giá vốn, v.v.).
        - Câu hỏi cảm tính hoặc kiến thức bên ngoài không suy ra được từ dữ liệu.
        Trả về (True, lý do) nếu chắc chắn 100% không trả lời được.
        """
        q = question_obj.get("question", "").lower()
        bid = question_obj.get("business_id", "")
        as_of = question_obj.get("as_of_date", "")

        # 1. Kiểm tra ngày ngoài khoảng dữ liệu của shop (chỉ áp dụng cho kỳ bán hàng, không áp dụng cho hạn dùng trong tương lai)
        is_expiry_q = any(k in q for k in ["hạn dùng", "han dung", "hạn sử dụng", "han su dung", "hsd", "hết hạn trong", "het han trong", "hết hạn vào", "het han vao"])
        if not is_expiry_q:
            s_min, s_max = store.get_shop_date_range(bid)
            s_max_effective = max(s_max, as_of) if (as_of and s_max) else (s_max or as_of)
            d_ctx = date_resolver.resolve(question_obj.get("question", ""), as_of)
            for p in d_ctx.get("detected_periods", []):
                st = p.get("start_date")
                en = p.get("end_date")
                if st and en:
                    if en < s_min or (s_max_effective and st > s_max_effective):
                        return True, f"Kỳ hỏi ({st} đến {en}) nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop ({s_min} đến {s_max_effective})."

        tables = store.load_tables(bid)
        all_cols = {c.lower() for df in tables.values() for c in df.columns}
        all_tbl_names = {t.lower() for t in tables.keys()}

        # 2. Câu hỏi cảm tính / thái độ / kỹ năng
        if any(k in q for k in ["tận tâm", "nhiệt tình", "thái độ", "tốt bụng", "lễ phép", "chu đáo", "tay nghề"]):
            return True, "Dữ liệu shop không có cột đánh giá cảm tính, thái độ hoặc tay nghề."

        # 3. Kiến thức y khoa / liều dùng ngoài dữ liệu
        if any(k in q for k in ["uống mỗi lần", "liều dùng", "cách dùng", "uống bao nhiêu viên", "uống mấy gói"]):
            return True, "Liều dùng y tế là kiến thức bên ngoài, không ghi trong dữ liệu danh mục hàng."

        # 4. Lý do hủy đơn hàng
        if any(k in q for k in ["vì giao trễ", "vi giao tre", "lý do hủy", "lý do huỷ", "lí do hủy", "lí do huỷ", "ly do huy", "li do"]):
            if not any(any(k in c for k in ["lý do", "lí do", "nguyên nhân", "ly_do", "reason"]) for c in all_cols):
                return True, "Shop không có cột ghi nhận lý do hủy/trả đơn hàng."

        # 5. Địa chỉ / đường phố của khách hàng
        sugar_words = ["đường đen", "đường phèn", "đường cát", "đường mía", "đường thốt nốt", "duong den", "duong phen", "duong cat"]
        q_no_sugar = q
        for sw in sugar_words:
            q_no_sugar = q_no_sugar.replace(sw, "sugar")
        if any(k in q_no_sugar for k in ["khách ở đường", "khách đường", "ở đường ", "tại đường ", "khach o duong", "o duong "]):
            if not any(any(k in c for k in ["địa chỉ", "dia_chi", "đường", "duong", "address"]) for c in all_cols):
                return True, "Shop không có thông tin địa chỉ hoặc tên đường của khách hàng."

        # 6. Giới tính khách hàng
        if any(k in q for k in ["khách nam", "khách nữ", "giới tính", "nam hay nữ", "khach nam", "khach nu", "gioi tinh"]):
            if not any(any(k in c for k in ["giới tính", "gioi_tinh", "gender"]) for c in all_cols):
                return True, "Shop không có cột ghi nhận giới tính của khách hàng."

        # 7. Lãi gộp / lãi ròng khi shop không có cột giá vốn
        if any(k in q for k in ["lãi gộp", "lai gop", "lợi nhuận gộp", "lãi ròng", "lai rong"]):
            has_cogs = any(any(k in c for k in ["giá vốn", "gia_von", "giavon", "cost", "cogs", "giávốn"]) for c in all_cols)
            if not has_cogs:
                return True, "Shop không có cột giá vốn trên dòng bán để tính lãi gộp theo Mục 5 Rulebook."

        # 8. Tiền nhập hàng khi shop không có bảng/cột nhập hàng
        if any(k in q for k in ["tiền nhập hàng", "tổng tiền nhập", "giá nhập bình quân", "tien nhap hang", "gia nhap"]):
            has_nhap = any("nhap" in t for t in all_tbl_names) or any("nhập" in c for c in all_cols)
            if not has_nhap:
                return True, "Shop không có dữ liệu nhập hàng."

        # 9. Hạn sử dụng (HSD) khi mặt hàng không có cột hạn sử dụng
        if any(k in q for k in ["hạn dùng", "han dung", "hạn sử dụng", "han su dung", "hsd"]):
            has_exp = any(any(k in c for k in ["hạn sử dụng", "hạn dùng", "hansudung", "hsd", "expiry", "exp_date"]) for c in all_cols)
            if not has_exp:
                return True, "Shop không có thông tin hạn sử dụng cho mặt hàng này."

        # 10. Tồn kho khi shop không có bảng tồn kho và không có dữ liệu nhập hàng
        if re.search(r"(?:trong\s*kho|tồn\s*kho|ton\s*kho|kho\s*còn|còn\s+.*(?:trong\s*kho|kho)|còn\s+bao\s+nhiêu\s+cái)", q):
            has_stock_tbl = any(any(k in t for k in ["ton_kho", "tonkho", "tồn kho", "tồn_kho", "nhap", "nhập"]) for t in all_tbl_names)
            has_stock_col = any(any(k in c for k in ["tồn cuối", "tồn thực", "ton kho", "tồn kho", "sl nhập", "số lượng nhập"]) for c in all_cols)
            if not (has_stock_tbl or has_stock_col):
                return True, "Shop chỉ có dữ liệu bán hàng, không có bảng tồn kho hoặc dữ liệu nhập hàng để tính tồn kho."

        return False, ""

    def select_relevant_rules(self, question: str, biz_readme: str = "") -> str:
        """
        Trích xuất các điều luật trong `rulebook.md` khớp với ngữ cảnh câu hỏi và ngành của shop.
        """
        q = question.lower()
        readme_lower = biz_readme.lower()

        selected_keys = []

        # Mục 1 (Đơn hàng & doanh thu) luôn quan trọng cho hầu hết truy vấn
        selected_keys.append("1. Đơn hàng và doanh thu")

        # Mục 3: Xếp hạng / bán chạy / trung bình đơn / top khách hàng
        if any(
            kw in q
            for kw in [
                "bán chạy",
                "chạy nhất",
                "chay nhat",
                "nhiều nhất",
                "nhieu nhat",
                "hot nhất",
                "lời nhất",
                "xếp hạng",
                "top",
                "nhân viên",
                "trung bình",
                "bình quân",
                "nhóm hàng",
                "phí phục vụ",
                "top khách",
                "khách mua nhiều",
                "khách quen",
                "hạng khách",
            ]
        ) or "phí phục vụ" in readme_lower:
            selected_keys.append("3. Xếp hạng")

        # Mục 4: Công nợ
        if any(kw in q for kw in ["nợ", "no ", "công nợ", "cong no", "trả nợ", "thu nợ", "chưa trả", "còn lại"]):
            selected_keys.append("4. Công nợ")

        # Mục 5: Kho, nhập hàng, giá vốn, lãi gộp, lãi ròng, bảng giá
        has_sec5 = any(
            kw in q
            for kw in [
                "nhập",
                "nhap",
                "tồn",
                "ton ",
                "kho",
                "giá vốn",
                "gia von",
                "lãi",
                "lai ",
                "hết hàng",
                "hàng ế",
                "bán ế",
                "bảng giá",
                "tăng giá",
                "đơn giá",
                "giá bán",
                "thùng",
                "hộp",
                "lốc",
            ]
        ) or bool(re.search(r"(?<!trả\s)(?<!hãy\s)\b(?:lời|tiền lời|sinh lời|có lời|hàng ế|tồn ế)\b", q))
        if has_sec5:
            selected_keys.append("5. Kho, nhập hàng, giá vốn")

        # Mục 6: Theo ngành (Tìm động các tiểu mục ngành khớp với README hoặc câu hỏi, không hardcode)
        industry_sections = [k for k in self.sections if k.startswith("6. Theo ngành - ")]
        for ind_key in industry_sections:
            ind_name = ind_key.replace("6. Theo ngành - ", "").strip()
            ind_clean = strip_accents(ind_name)
            # 1. So khớp tên ngành trực tiếp trong câu hỏi hoặc README
            if ind_name.lower() in q:
                selected_keys.append(ind_key)
                continue
            if ind_name.lower() in readme_lower or ind_clean in strip_accents(biz_readme):
                selected_keys.append(ind_key)
                continue
            # 2. Hoặc so khớp từ khóa đặc trưng trong nội dung tiểu mục luật ngành đó
            sec_content = self.sections.get(ind_key, "")
            bold_terms = re.findall(r"\*\*([^\*]+)\*\*", sec_content)
            keywords_in_sec = [t.lower().strip() for t in bold_terms if len(t.strip()) >= 3 and not t.strip().startswith("Mục")]
            if any(term in q or term in readme_lower for term in keywords_in_sec):
                selected_keys.append(ind_key)

        # Tóm tắt cô đọng Mục 2 (Thời gian & Ranh giới) dựa trên núm đọc động từ rulebook
        morning_str = f"{self.knobs.shift_morning[0]}–{self.knobs.shift_morning[1]}"
        afternoon_str = f"{self.knobs.shift_afternoon[0]}–{self.knobs.shift_afternoon[1]}"
        compact_footer = (
            f"## Quy tắc Thời gian & Ranh giới (Mục 2):\n"
            f"- Tuần bắt đầu từ: {self.knobs.week_start_day}.\n"
            f"- 'Quá N, hơn N, trên N, dưới N': không gồm N; 'Từ N trở lên, từ N trở xuống': gồm N.\n"
            f"- Ca sáng: {morning_str}, ca chiều: {afternoon_str} (so ca chiều với ca sáng: mốc là ca sáng).\n"
            f"- Hỏi số còn lại ('tính đến hết ngày X', 'còn nợ', 'còn tồn'): tính mọi phát sinh từ đầu dữ liệu đến hết ngày đó."
        )

        seen = set()
        scored_blocks: List[Tuple[int, str]] = []
        for k in selected_keys:
            if k in seen:
                continue
            seen.add(k)
            for sec_title, sec_body in self.sections.items():
                if sec_title == k or sec_title.startswith(k):
                    prio = 10
                    k_low = k.lower()
                    if "4. công nợ" in k_low and any(w in q for w in ["nợ", "công nợ"]):
                        prio = 50
                    elif "5. kho" in k_low and any(w in q for w in ["tồn", "kho", "nhập"]):
                        prio = 40
                    elif "3. xếp hạng" in k_low and any(w in q for w in ["top", "chạy nhất", "nhiều nhất", "xếp hạng"]):
                        prio = 30
                    elif "1. đơn hàng" in k_low:
                        prio = 20
                    elif k_low.startswith("6. theo ngành") and any(w in q for w in ["thuốc", "vật liệu", "bàn", "món", "ly"]):
                        prio = 35
                    scored_blocks.append((prio, sec_body))
                    break

        # Giới hạn trần rules_prompt <= 2500 ký tự (ưu tiên section có điểm liên quan trực tiếp cao nhất)
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        kept_blocks: List[str] = []
        c_len = 0
        for _, body in scored_blocks:
            if c_len + len(body) <= 2500 or not kept_blocks:
                kept_blocks.append(body)
                c_len += len(body)

        kept_blocks.append(compact_footer)
        return "\n\n".join(kept_blocks)

    @staticmethod
    def extract_readme_max_date(biz_readme: str) -> Optional[date]:
        """Trích xuất ngày cuối dữ liệu từ dòng 'Dữ liệu có đến hết DD/MM/YYYY' trong README của shop."""
        m = re.search(r"Dữ liệu có đến hết\s+(\d{1,2})/(\d{1,2})/(20\d{2})", biz_readme)
        if not m:
            return None
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None


class DAILFewShotSelector:
    """
    Kỹ thuật Few-Shot In-Context Learning theo kiến trúc DAIL-SQL (Alibaba Group, PVLDB/arXiv:2308.15363).
    - DAIL-O (DAIL Organization): Giữ cấu trúc Question -> SQL mẫu, loại bỏ schema lặp lại của DB ví dụ để tiết kiệm token tối đa.
    - DAIL-S (DAIL Selection): Nhận diện khung xương truy vấn (Query Skeleton Intent) để chọn đúng 1-2 ví dụ chuẩn nhất.
    """

    EXEMPLAR_CANCEL_RATE = (
        "/* Ví dụ DAIL-O 1: Tính tỉ lệ đơn huỷ (%) trong kỳ */\n"
        "-- Câu hỏi: Trong tháng 5/2025, tỉ lệ đơn huỷ là bao nhiêu phần trăm?\n"
        "-- Lưu ý: BẮT BUỘC dùng COUNT(DISTINCT ...) cho cả tử số (đơn hủy) và mẫu số (tổng đơn). Không đếm số dòng sản phẩm:\n"
        "SELECT ROUND(\n"
        "  COUNT(DISTINCT CASE WHEN \"status_norm\" = 'Đã hủy' THEN \"Mã đơn\" END) * 100.0 / \n"
        "  NULLIF(COUNT(DISTINCT \"Mã đơn\"), 0), \n"
        "  2\n"
        ") AS answer\n"
        "FROM don_hang\n"
        "WHERE \"Thời gian\" >= '2025-05-01' AND \"Thời gian\" < '2025-06-01';"
    )

    EXEMPLAR_GROWTH_PREV_MONTH = (
        "/* Ví dụ DAIL-O 2: So sánh doanh thu với tháng liền trước (%) */\n"
        "-- Câu hỏi: So với tháng liền trước, doanh thu tháng 6 năm 2026 thay đổi bao nhiêu %?\n"
        "-- Lưu ý: Tháng liền trước của T6/2026 là cả tháng 5/2026 (M-1). Tuyệt đối KHÔNG so với tháng 6/2025 (cùng kỳ năm trước):\n"
        "WITH monthly AS (\n"
        "  SELECT SUBSTR(\"Thời gian\", 1, 7) AS ym, SUM(\"revenue_norm\") AS rev\n"
        "  FROM ban_hang\n"
        "  WHERE \"status_norm\" = 'Hoàn thành' AND SUBSTR(\"Thời gian\", 1, 7) IN ('2026-05', '2026-06')\n"
        "  GROUP BY ym\n"
        ")\n"
        "SELECT ROUND((curr.rev - prev.rev) * 100.0 / NULLIF(prev.rev, 0), 2) AS answer\n"
        "FROM (SELECT rev FROM monthly WHERE ym = '2026-06') curr,\n"
        "     (SELECT rev FROM monthly WHERE ym = '2026-05') prev;"
    )

    EXEMPLAR_GROWTH_SAME_PERIOD_MONTH = (
        "/* Ví dụ DAIL-O 3: So sánh doanh thu với cùng kỳ tháng trước (%) */\n"
        "-- Câu hỏi: Từ đầu tháng này đến hết hôm nay (15/07/2026), doanh thu thay đổi bao nhiêu % so với cùng kỳ tháng trước?\n"
        "-- Lưu ý: Cùng kỳ tháng trước là từ ngày 01 đến cùng ngày D của tháng trước (01/06/2026 đến 15/06/2026):\n"
        "WITH p AS (\n"
        "  SELECT \n"
        "    SUM(CASE WHEN \"Thời gian\" >= '2026-07-01' AND \"Thời gian\" <= '2026-07-15 23:59:59' THEN \"revenue_norm\" ELSE 0 END) AS curr_rev,\n"
        "    SUM(CASE WHEN \"Thời gian\" >= '2026-06-01' AND \"Thời gian\" <= '2026-06-15 23:59:59' THEN \"revenue_norm\" ELSE 0 END) AS prev_rev\n"
        "  FROM ban_hang\n"
        "  WHERE \"status_norm\" = 'Hoàn thành'\n"
        ")\n"
        "SELECT ROUND((curr_rev - prev_rev) * 100.0 / NULLIF(prev_rev, 0), 2) AS answer FROM p;"
    )

    EXEMPLAR_GROWTH_SAME_PERIOD_YEAR = (
        "/* Ví dụ DAIL-O 4: So sánh doanh thu với cùng kỳ năm trước (%) */\n"
        "-- Câu hỏi: Doanh thu tháng 5/2026 tăng trưởng bao nhiêu % so với cùng kỳ năm trước (tháng 5/2025)?\n"
        "-- Lưu ý: So cùng kỳ năm trước là so với cùng tháng M của năm Y-1:\n"
        "WITH monthly AS (\n"
        "  SELECT SUBSTR(\"Thời gian\", 1, 7) AS ym, SUM(\"revenue_norm\") AS rev\n"
        "  FROM ban_hang\n"
        "  WHERE \"status_norm\" = 'Hoàn thành' AND SUBSTR(\"Thời gian\", 1, 7) IN ('2025-05', '2026-05')\n"
        "  GROUP BY ym\n"
        ")\n"
        "SELECT ROUND((curr.rev - prev.rev) * 100.0 / NULLIF(prev.rev, 0), 2) AS answer\n"
        "FROM (SELECT rev FROM monthly WHERE ym = '2026-05') curr,\n"
        "     (SELECT rev FROM monthly WHERE ym = '2025-05') prev;"
    )

    EXEMPLAR_TOP_ITEMS_QTY_BASE = (
        "/* Ví dụ DAIL-O 5: Tìm mặt hàng bán chạy nhất */\n"
        "-- Câu hỏi: Hàng nào bán chạy nhất tháng 1/2025?\n"
        "-- Lưu ý: BẮT BUỘC ORDER BY SUM(\"qty_base\") DESC (theo số lượng quy đổi ra đơn vị cơ bản lẻ nhỏ nhất). Không order by số lượng thô:\n"
        "SELECT \"Tên hàng\" AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND \"Thời gian\" >= '2025-01-01' AND \"Thời gian\" < '2025-02-01'\n"
        "GROUP BY \"Tên hàng\"\n"
        "ORDER BY SUM(COALESCE(\"qty_base\", \"Số lượng\")) DESC\n"
        "LIMIT 1;"
    )

    EXEMPLAR_SINGLE_ITEM_QUERY = (
        "/* Ví dụ DAIL-O 6: Doanh thu của một món ăn / mặt hàng cụ thể */\n"
        "-- Câu hỏi: Xem giúp tiền bán cơm chiên dương châu tháng 3/2025 là bao nhiêu?\n"
        "-- Lưu ý: BẮT BUỘC lọc đúng tên món trong mệnh đề WHERE (ILIKE). Tuyệt đối KHÔNG tính tổng cả tháng của toàn cửa hàng:\n"
        "SELECT COALESCE(SUM(\"revenue_norm\"), 0) AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND \"Thời gian\" >= '2025-03-01' AND \"Thời gian\" < '2025-04-01'\n"
        "  AND \"Tên hàng\" ILIKE '%cơm chiên dương châu%';"
    )

    EXEMPLAR_CUSTOMER_DEBT = (
        "/* Ví dụ DAIL-O 7: Bảng khách hàng có công nợ lớn nhất */\n"
        "-- Câu hỏi: Top 3 khách có công nợ lớn nhất đến hết ngày 14/09/2026?\n"
        "-- Lưu ý: Bắt buộc loại trừ 'Khách lẻ' / 'khách vãng lai' và các dòng nợ <= 0:\n"
        "SELECT \"Tên khách\" AS khach_hang, SUM(\"Nợ\") AS so_tien_no\n"
        "FROM cong_no\n"
        "WHERE \"Thời gian\" <= '2026-09-14 23:59:59'\n"
        "  AND \"Tên khách\" NOT ILIKE '%khách lẻ%' AND \"Tên khách\" NOT ILIKE '%lẻ%' AND \"Tên khách\" IS NOT NULL\n"
        "GROUP BY \"Tên khách\"\n"
        "HAVING SUM(\"Nợ\") > 0\n"
        "ORDER BY so_tien_no DESC\n"
        "LIMIT 3;"
    )

    EXEMPLAR_SHIFT_COMPARISON = (
        "/* Ví dụ DAIL-O 8: So sánh ca chiều với ca sáng (%) */\n"
        "-- Câu hỏi: Tháng 12/2025 doanh thu ca chiều tăng hay giảm bao nhiêu % so với ca sáng?\n"
        "-- Lưu ý: Ca sáng: 06:00-13:59:59; Ca chiều: 14:00-21:59:59. Mẫu số so sánh là Ca sáng:\n"
        "WITH shifts AS (\n"
        "  SELECT \n"
        "    SUM(CASE WHEN hour(try_cast(\"Thời gian\" AS TIMESTAMP)) BETWEEN 6 AND 13 THEN \"revenue_norm\" ELSE 0 END) AS sang,\n"
        "    SUM(CASE WHEN hour(try_cast(\"Thời gian\" AS TIMESTAMP)) BETWEEN 14 AND 21 THEN \"revenue_norm\" ELSE 0 END) AS chieu\n"
        "  FROM ban_hang\n"
        "  WHERE \"status_norm\" = 'Hoàn thành'\n"
        "    AND \"Thời gian\" >= '2025-12-01' AND \"Thời gian\" < '2026-01-01'\n"
        ")\n"
        "SELECT ROUND((chieu - sang) * 100.0 / NULLIF(sang, 0), 2) AS answer FROM shifts;"
    )

    EXEMPLAR_DISTINCT_CUSTOMERS = (
        "/* Ví dụ DAIL-O 9: Đếm số lượng khách hàng mua hàng (loại bỏ Khách lẻ) */\n"
        "-- Câu hỏi: Trong tháng 8/2025 có bao nhiêu khách hàng khác nhau đến mua hàng?\n"
        "-- Lưu ý: Dùng COUNT(DISTINCT ...) và bắt buộc loại trừ 'Khách lẻ':\n"
        "SELECT COUNT(DISTINCT \"Khách hàng\") AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND CAST(\"Ngày\" AS DATE) BETWEEN DATE '2025-08-01' AND DATE '2025-08-31'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%khách lẻ%' AND \"Khách hàng\" IS NOT NULL;"
    )

    EXEMPLAR_SALES_BY_CHANNEL = (
        "/* Ví dụ DAIL-O 10: Doanh thu theo từng kênh bán hàng hoặc hình thức thanh toán */\n"
        "-- Câu hỏi: Doanh thu các đơn bán tại quầy và các đơn giao tận nơi trong tháng này mỗi loại bao nhiêu?\n"
        "-- Lưu ý: Trả về dạng list [tien_quay, tien_giao] theo thứ tự hỏi:\n"
        "SELECT [\n"
        "  COALESCE(SUM(CASE WHEN \"Kênh bán\" ILIKE '%quầy%' THEN \"revenue_norm\" ELSE 0 END), 0),\n"
        "  COALESCE(SUM(CASE WHEN \"Kênh bán\" ILIKE '%giao%' OR \"Kênh bán\" ILIKE '%ship%' THEN \"revenue_norm\" ELSE 0 END), 0)\n"
        "] AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND CAST(\"Ngày\" AS DATE) BETWEEN DATE 'start_date' AND DATE 'end_date';"
    )

    EXEMPLAR_YTD_REVENUE = (
        "/* Ví dụ DAIL-O 11: Doanh thu lũy kế từ đầu năm đến nay (Year-To-Date) */\n"
        "-- Câu hỏi: Tính từ đầu năm đến hết ngày hôm nay (15/09/2026), tổng doanh thu là bao nhiêu?\n"
        "-- Lưu ý: Lọc từ ngày 01/01 của năm đó đến as_of_date:\n"
        "SELECT COALESCE(SUM(\"revenue_norm\"), 0) AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND CAST(\"Ngày\" AS DATE) BETWEEN DATE '2026-01-01' AND DATE '2026-09-15';"
    )

    EXEMPLAR_EXPIRY_GOODS = (
        "/* Ví dụ DAIL-O 12: Đếm số mặt hàng cận date hoặc hết hạn sử dụng (HSD) */\n"
        "-- Câu hỏi: Tính đến hôm nay, có mấy lô hàng đã hết hạn sử dụng?\n"
        "-- Lưu ý: So sánh cột Hạn sử dụng với ngày hiện tại:\n"
        "SELECT COUNT(*) AS answer\n"
        "FROM ton_kho\n"
        "WHERE CAST(\"Hạn sử dụng\" AS DATE) <= DATE '2025-06-30';"
    )

    EXEMPLAR_TOP_CUSTOMER = (
        "/* Ví dụ DAIL-O 13: Tìm khách hàng mua nhiều nhất / chi tiêu nhiều nhất */\n"
        "-- Câu hỏi: Khách hàng nào chi tiêu nhiều nhất trong tháng 5/2025?\n"
        "-- Lưu ý: BẮT BUỘC loại bỏ 'Khách lẻ', 'lẻ', 'vãng lai' và NULL để tìm tên khách hàng cụ thể:\n"
        "SELECT \"Khách hàng\" AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND \"Thời gian\" >= '2025-05-01' AND \"Thời gian\" < '2025-06-01'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%khách lẻ%'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%lẻ%'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%vãng lai%'\n"
        "  AND \"Khách hàng\" IS NOT NULL\n"
        "GROUP BY \"Khách hàng\"\n"
        "ORDER BY SUM(\"revenue_norm\") DESC\n"
        "LIMIT 1;"
    )

    EXEMPLAR_CATEGORY_SHARE = (
        "/* Ví dụ DAIL-O 14: Tỉ trọng đóng góp doanh thu của một nhóm hàng / kênh (%) */\n"
        "-- Câu hỏi: Nhóm hàng Nước uống đóng góp bao nhiêu % trong tổng doanh thu tháng 7/2025?\n"
        "-- Lưu ý: Tử số là doanh thu nhóm hàng, mẫu số là tổng doanh thu toàn cửa hàng trong cùng kỳ:\n"
        "SELECT ROUND(\n"
        "  COALESCE(SUM(CASE WHEN \"Nhóm hàng\" ILIKE '%nước uống%' THEN \"revenue_norm\" ELSE 0 END), 0) * 100.0 /\n"
        "  NULLIF(SUM(\"revenue_norm\"), 0),\n"
        "  2\n"
        ") AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND \"Thời gian\" >= '2025-07-01' AND \"Thời gian\" < '2025-08-01';"
    )

    EXEMPLAR_TOP_CUSTOMERS_LIST = (
        "/* Ví dụ DAIL-O 15: Top N khách hàng mua nhiều nhất / chi tiêu nhiều nhất (dạng danh sách) */\n"
        "-- Câu hỏi: 5 khách hàng có tổng tiền mua cao nhất tuần rồi là ai?\n"
        "-- Lưu ý: BẮT BUỘC loại bỏ 'Khách lẻ', 'lẻ', 'vãng lai' và NULL trong WHERE TRƯỚC KHI LIMIT N để trả về đủ N khách hàng thực tế:\n"
        "SELECT \"Khách hàng\"\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND CAST(\"Ngày\" AS DATE) BETWEEN DATE '2025-07-28' AND DATE '2025-08-03'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%khách lẻ%'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%lẻ%'\n"
        "  AND \"Khách hàng\" NOT ILIKE '%vãng lai%'\n"
        "  AND \"Khách hàng\" IS NOT NULL\n"
        "GROUP BY \"Khách hàng\"\n"
        "ORDER BY SUM(COALESCE(\"revenue_norm\", \"Thành tiền\")) DESC\n"
        "LIMIT 5;"
    )

    EXEMPLAR_SINGLE_ITEM_QTY = (
        "/* Ví dụ DAIL-O 16: Số lượng bán ra của một món ăn / mặt hàng cụ thể (mấy đĩa / kg / ly / lon / cái / nồi) */\n"
        "-- Câu hỏi: Trong tháng 1/2025 cửa hàng bán được mấy nồi lẩu gà lá é?\n"
        "-- Lưu ý: Hỏi SỐ LƯỢNG (mấy nồi, mấy đĩa, mấy kg, mấy lon, bao nhiêu gói) thì BẮT BUỘC tính tổng sản lượng `qty_base` (hoặc Số lượng), tuyệt đối KHÔNG tính doanh thu tiền:\n"
        "SELECT COALESCE(SUM(COALESCE(\"qty_base\", \"Số lượng\")), 0) AS answer\n"
        "FROM ban_hang\n"
        "WHERE \"status_norm\" = 'Hoàn thành'\n"
        "  AND \"Thời gian\" >= '2025-01-01' AND \"Thời gian\" < '2025-02-01'\n"
        "  AND \"Tên hàng\" ILIKE '%lẩu gà lá é%';"
    )

    EXEMPLAR_BOOLEAN_TREND = (
        "/* Ví dụ DAIL-O 17: Kiểm tra xu hướng doanh thu có tăng liên tục qua các tháng không (True/False) */\n"
        "-- Câu hỏi: Doanh thu 3 tháng 5/2025, 6/2025, 7/2025 có tăng liên tục không?\n"
        "-- Lưu ý: Tính doanh thu từng tháng bằng WITH monthly AS (...), sau đó so sánh bằng scalar subqueries:\n"
        "WITH monthly AS (\n"
        "  SELECT SUBSTR(\"Thời gian\", 1, 7) AS ym, SUM(\"revenue_norm\") AS rev\n"
        "  FROM ban_hang\n"
        "  WHERE \"status_norm\" = 'Hoàn thành' AND SUBSTR(\"Thời gian\", 1, 7) IN ('2025-05', '2025-06', '2025-07')\n"
        "  GROUP BY ym\n"
        ")\n"
        "SELECT CASE\n"
        "  WHEN (SELECT rev FROM monthly WHERE ym = '2025-05') < (SELECT rev FROM monthly WHERE ym = '2025-06')\n"
        "   AND (SELECT rev FROM monthly WHERE ym = '2025-06') < (SELECT rev FROM monthly WHERE ym = '2025-07')\n"
        "  THEN TRUE ELSE FALSE\n"
        "END AS answer;"
    )

    @classmethod
    def select_exemplars(cls, question: str, has_entity_hint: bool = False) -> str:
        """
        DAIL-S: Tự động trích xuất 1-2 ví dụ có khung xương (Skeleton) tương đồng nhất theo intent của câu hỏi.
        """
        q = strip_accents(question).lower().strip()
        selected: List[str] = []

        # Intent 1: Tỉ lệ đơn hủy
        if any(kw in q for kw in ["ti le huy", "ti le don huy", "% don bi huy", "% don huy", "don bi huy"]):
            selected.append(cls.EXEMPLAR_CANCEL_RATE)

        # Intent 2: So sánh cùng kỳ tháng trước
        if any(kw in q for kw in ["cung ky thang truoc", "cung ky th truoc", "cung ky thang trc", "cung ky th trc"]):
            selected.append(cls.EXEMPLAR_GROWTH_SAME_PERIOD_MONTH)

        # Intent 3: So sánh tháng liền trước (không phải cùng kỳ)
        elif any(kw in q for kw in ["thang lien truoc", "thang truoc do", "hon hay kem thang truoc", "hon hay kem thang trc", "so voi thang truoc", "so voi thang trc"]):
            selected.append(cls.EXEMPLAR_GROWTH_PREV_MONTH)

        # Intent 4: So sánh cùng kỳ năm trước
        elif any(kw in q for kw in ["cung ky nam truoc", "cung ky nam ngoai", "so voi cung ky nam truoc"]):
            selected.append(cls.EXEMPLAR_GROWTH_SAME_PERIOD_YEAR)

        # Intent 5: Tỉ trọng đóng góp danh mục / nhóm hàng (%)
        if any(kw in q for kw in ["dong gop", "chiem bao nhieu %", "chiem bao nhieu phan tram", "ti trong", "ty trong", "phan tram doanh thu"]):
            selected.append(cls.EXEMPLAR_CATEGORY_SHARE)

        # Intent 6: Top khách hàng mua nhiều nhất / chi tiêu nhiều nhất (loại bỏ Khách lẻ)
        if any(kw in q for kw in [
            "ai mua", "khach nao mua", "khach nao chi", "khach hang nao mua", "khach mua nhieu",
            "chi tieu nhieu nhat", "mua nhieu nhat", "ai chi tieu", "ai la nguoi mua",
            "mua cao nhat", "tong tien mua cao", "tien mua cao nhat", "chi nhieu nhat", "mua nhieu tien nhat"
        ]):
            if any(kw in q for kw in ["top 3", "top 5", "top 10", "3 khach", "5 khach", "nhung khach"]):
                selected.append(cls.EXEMPLAR_TOP_CUSTOMERS_LIST)
            else:
                selected.append(cls.EXEMPLAR_TOP_CUSTOMER)

        # Intent 7: So sánh ca chiều với ca sáng
        if any(kw in q for kw in ["ca chieu", "ca sang"]) and any(kw in q for kw in ["so voi", "so vs", "chenh", "tang hay giam"]):
            selected.append(cls.EXEMPLAR_SHIFT_COMPARISON)

        # Intent 8: Top bán chạy / chạy nhất
        if any(kw in q for kw in ["chay nhat", "hot nhat", "ban nhieu nhat", "ban chay nhat", "top 3 sp", "top 5 sp", "top 5 hang"]):
            selected.append(cls.EXEMPLAR_TOP_ITEMS_QTY_BASE)

        # Intent 9: Đếm số lượng khách hàng khác nhau
        if any(kw in q for kw in ["khach hang khac nhau", "so khach den mua", "may khach den mua", "may khach mua", "bao nhieu khach hang den mua"]):
            selected.append(cls.EXEMPLAR_DISTINCT_CUSTOMERS)

        # Intent 10: Doanh thu theo kênh bán / hình thức
        if any(kw in q for kw in ["moi loai bao nhieu", "moi hinh thuc bao nhieu", "ban tai quay va"]) and any(kw in q for kw in ["doanh thu", "dthu"]):
            selected.append(cls.EXEMPLAR_SALES_BY_CHANNEL)

        # Intent 11: Doanh thu lũy kế từ đầu năm (YTD)
        if any(kw in q for kw in ["tu dau nam", "dau nam den nay", "luy ke tu dau nam"]):
            selected.append(cls.EXEMPLAR_YTD_REVENUE)

        # Intent 12: Hết hạn sử dụng / HSD / Cận date
        if any(kw in q for kw in ["het han", "can date", "het date", "qua han dung"]):
            selected.append(cls.EXEMPLAR_EXPIRY_GOODS)

        # Intent 13: Xu hướng doanh thu có tăng liên tục không (Boolean)
        if any(kw in q for kw in ["tang lien tuc", "tăng liên tục", "luon tang", "luôn tăng", "tang deu", "tăng đều"]):
            selected.append(cls.EXEMPLAR_BOOLEAN_TREND)

        # Intent 14: Hỏi về 1 món ăn / sản phẩm cụ thể (Số lượng vs Doanh thu)
        if has_entity_hint and not any(kw in q for kw in ["chay nhat", "hot nhat", "ban nhieu nhat"]):
            is_qty_q = any(kw in q for kw in [
                "may noi", "may dia", "may kg", "may ly", "may lon", "may chai", "may hop", "may goi",
                "so kg", "so luong kg", "bao nhieu noi", "bao nhieu dia", "bao nhieu ly", "bao nhieu kg",
                "bao nhieu lon", "bao nhieu chai", "bao nhieu hop", "bao nhieu goi", "san luong", "so luong",
            ])
            if is_qty_q:
                selected.append(cls.EXEMPLAR_SINGLE_ITEM_QTY)
            else:
                selected.append(cls.EXEMPLAR_SINGLE_ITEM_QUERY)

        # Intent 15: Công nợ khách hàng
        if any(kw in q for kw in ["cong no", "con no", "no nhieu nhat"]):
            selected.append(cls.EXEMPLAR_CUSTOMER_DEBT)

        # Giới hạn tối đa 2 ví dụ để đảm bảo token efficiency theo DAIL-O
        chosen = selected[:2]
        if not chosen:
            # Fallback ví dụ doanh thu thuần cơ bản
            return ""

        formatted_block = (
            "=== VÍ DỤ MINH HỌA TRUY VẤN DUCKDB SQL CHUẨN (DAIL-O FEW-SHOT) ===\n"
            + "\n\n".join(chosen)
            + "\n"
        )
        return formatted_block

