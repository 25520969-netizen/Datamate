# -*- coding: utf-8 -*-
"""
Module 2B: Query Analyzer & Rule Knowledge Injector.
- Phân loại kiểu đáp án kỳ vọng (number, percent, date, boolean, string, list, table).
- Trích xuất đúng các mục luật liên quan từ `train/rulebook.md` và `README.md` của shop.
- Kiểm tra nhanh dấu hiệu câu hỏi không trả lời được (`unanswerable`) khi kỳ hỏi nằm hoàn toàn ngoài vùng dữ liệu.
"""

import re
import unicodedata
from functools import lru_cache
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.lexicon import load_lexicon
from src.vietnamese_normalizer import VietnameseNormalizer


@lru_cache(maxsize=65536)
def _strip_accents_str(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).lower().strip()


def strip_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt và đưa về chữ thường để so khớp thực thể mềm (có cache: hàm thuần, gọi lặp rất nhiều)."""
    if not isinstance(text, str):
        text = str(text)
    return _strip_accents_str(text)


@lru_cache(maxsize=None)
def _keyword_patterns(keyword: str):
    """Biên dịch một lần mỗi từ khoá (trước đây biên dịch lại mỗi câu hỏi, làm tràn cache regex của Python)."""
    k = keyword.strip()
    exact = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)")
    na = re.compile(r"(?<!\w)" + re.escape(strip_accents(k)) + r"(?!\w)") if len(k.split()) >= 2 else None
    return exact, na


def keyword_in_question(keyword: str, q: str, q_na: str) -> bool:
    """
    Từ khoá có trong câu hỏi, theo ranh giới từ. Cụm >= 2 từ so cả dạng không dấu (câu hỏi gõ không dấu);
    từ đơn chỉ so có dấu vì bỏ dấu làm trùng từ khác ("bão" ~ "bao", "tâm" ~ "tam").
    """
    exact, na = _keyword_patterns(keyword)
    if exact.search(q):
        return True
    return na is not None and na.search(q_na) is not None


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
        self._prepared: Dict[str, Any] = {}

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
            for nc in name_cols:
                # Cột tên có thể đồng thời trùng từ khoá đơn vị (vd "Hàng - Unit"): không dùng chính nó làm cột ĐVT
                # (trước đây tạo DataFrame trùng cột và làm lỗi mọi câu hỏi của shop đó)
                dvt_col = next((c for c in dvt_cols if c != nc), None)
                subset_cols = [nc] + ([dvt_col] if dvt_col else [])
                sub = df[subset_cols].drop_duplicates()
                names = sub[nc].tolist()
                dvts = sub[dvt_col].tolist() if dvt_col else [None] * len(names)
                for name_val, dvt_val in zip(names, dvts):
                    orig = str(name_val).strip()
                    if not orig or orig.lower() in ["none", "nan", "", "<na>", "null"]:
                        continue
                    dvt = str(dvt_val).strip() if dvt_col and pd.notna(dvt_val) else ""
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

    def ground(self, business_id: str, question: str, tables: Dict[str, Any]) -> str:
        """Alias cho find_grounding_hints."""
        return self.find_grounding_hints(question, business_id, tables)

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

        # Danh sách khoá đã sắp xếp và các từ của khoá được tính MỘT lần mỗi shop (trước đây mỗi câu hỏi sắp xếp lại
        # hàng nghìn khoá và biên dịch lại regex cho từng khoá)
        prepared = self._prepared.get(business_id)
        if prepared is None:
            keys_sorted = sorted(entity_map.keys(), key=len, reverse=True)
            key_tokens = {}
            for k in keys_sorted:
                ck = re.sub(r"\(.*?\)", "", k).strip()
                key_tokens[k] = [t for t in re.split(r"[\s_]+", ck) if t]
            prepared = (keys_sorted, key_tokens)
            self._prepared[business_id] = prepared
        sorted_keys, key_tokens = prepared

        for key in sorted_keys:
            # Tiền kiểm tra chuỗi con: regex bên dưới chỉ có thể khớp khi khoá xuất hiện nguyên văn trong câu hỏi
            if key in q_clean and re.search(r"(?:^|[^\w])" + re.escape(key) + r"(?:[^\w]|$)", q_clean):
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
                toks = key_tokens[key]
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


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "assets" / "rulebook.md"


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

    strip_accents = staticmethod(strip_accents)

    @classmethod
    def read_rulebook_knobs(cls, path: Any) -> Dict[str, Any]:
        if not path or not Path(path).exists():
            return RulebookKnobReader.parse_knobs("").__dict__
        text = Path(path).read_text(encoding="utf-8")
        return RulebookKnobReader.parse_knobs(text).__dict__

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

        # Tách mục theo tiêu đề "## ..."; mục con "### ..." của một mục "theo ngành" (bất kể số thứ tự) tách riêng
        current_header = "intro"
        parent_h2 = ""
        buf: List[str] = []
        for line in self.full_rulebook.splitlines():
            if line.startswith("## "):
                self.sections[current_header] = "\n".join(buf).strip()
                current_header = parent_h2 = line.lstrip("#").strip()
                buf = [line]
            elif line.startswith("### ") and "ngành" in parent_h2.lower():
                self.sections[current_header] = "\n".join(buf).strip()
                current_header = f"{parent_h2} - {line.lstrip('#').strip()}"
                buf = [line]
            else:
                buf.append(line)
        self.sections[current_header] = "\n".join(buf).strip()

    @property
    def walk_in_not_customer(self) -> bool:
        """Rulebook đang dùng có ghi khách lẻ / khách vãng lai không phải khách (không đếm, không xếp hạng) hay không."""
        if not self.full_rulebook:
            return True
        text = unicodedata.normalize("NFC", self.full_rulebook).lower()
        return bool(re.search(r"khách lẻ[^\n|]{0,40}(?:không phải khách|không đếm|không xếp hạng|không tính)", text))

    def _section_title(self, *keywords: str) -> Optional[str]:
        """Tìm mục rulebook theo từ khoá trong tiêu đề (không phụ thuộc số thứ tự mục)."""
        for title in self.sections:
            t = title.lower()
            if any(k in t for k in keywords):
                return title
        return None

    @staticmethod
    def normalize_telex_typos(s: str) -> str:
        """Chuẩn hóa Unicode NFC và viết tắt nghiệp vụ bằng VietnameseNormalizer (luật viết tay, không dùng ViLexNorm)."""
        return VietnameseNormalizer.clean_deterministic(s)

    @staticmethod
    def infer_expected_type(question: str) -> str:
        """
        Dự đoán kiểu đáp án kỳ vọng từ câu hỏi tiếng Việt (number, percent, date, boolean, string, list, table).
        Lưu ý: Đây là gợi ý cho LLM; kiểu cuối cùng có thể được code Python tự xác định hoặc chuẩn hóa.
        """
        q = RuleEngine.normalize_telex_typos(question.lower().strip())
        q_no_acc = strip_accents(q)

        # 1. Table: bảng / mỗi ... kèm số tiền / danh sách ... và doanh thu/số tiền/số lượng/công nợ
        if (
            re.search(
                r"\b(mỗi người còn nợ bao nhiêu|mỗi người nợ bao nhiêu|mỗi người nợ bn|mỗi khách nợ bao nhiêu|"
                r"mỗi người bao nhiêu tiền nợ|số nợ của từng người|mỗi người đang nợ bao nhiêu|"
                r"moi nguoi con no|moi nguoi no|tung nguoi con no|"
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
            r"tháng nào|thang nao|vào tháng nào|vao thang nao)\b",
            q,
        ):
            return "date"

        # 5. List:
        # a. Hỏi 2 hoặc nhiều giá trị: "mỗi loại bao nhiêu", "lần lượt là bao nhiêu", "theo thứ tự đó", "mấy ... chẵn ... lẻ"
        # hoặc nhiều vế câu hỏi bao nhiêu, hoặc 2 doanh thu nối bằng "và" / "va"
        list_kw_regex = (
            r"\b(mỗi loại bao nhiêu|mỗi loại bn|mỗi bên bao nhiêu|mỗi bên bn|mỗi thứ bao nhiêu|mỗi thứ bn|"
            r"moi loai bnhiu|moi loai bn|moi loai bao nhieu|moi ben bn|moi ben bao nhieu|"
            r"mỗi tháng là bao nhiêu|mỗi tháng là bnhiu|mỗi tháng bnhiu|moi thang la bao nhieu|moi thang la bnhiu|"
            r"lần lượt là bao nhiêu|lần lượt là bn|lần lượt theo thứ tự|theo thứ tự đó|theo thứ tự thời gian|theo thu tu thoi gian|"
            r"ghi (?:hai|ba|[2-9]) số|ghi (?:hai|ba|[2-9]) so|hai số|2 số|ba số|3 số|trả lời hai số|tra loi hai so|trả lời 2 số|tra loi 2 so|"
            r"ghi lần lượt|ghi lan luot|"
            r"chẵn.*lẻ|chan.*le|hộp chẵn|vỉ chẵn|thùng chẵn|"
            r"từ tháng cũ đến tháng mới|tu thang cu den thang moi)\b"
        )
        if (
            re.search(list_kw_regex, q)
            or re.search(list_kw_regex, q_no_acc)
            or len(re.findall(r"\b(?:bao\s*(?:nhiêu|nhieu|nhiu)|bn|bnhiu)\b", q_no_acc)) >= 2
            or re.search(r"(?:dthu|doanh thu|tiền|tien).*?\b(?:và|va)\b.*?(?:dthu|doanh thu|tiền|tien)", q_no_acc)
            or re.search(r"(?:dthu|doanh thu|tiền|tien).*?[,;]\s*(?:doanh thu|dthu|tiền|tien)", q_no_acc)
            or re.search(r"(?:ghi\s*)?nợ\s*thêm.*?trả\s*(?:nợ\s*)?(?:được|dc|duoc)", q)
            or re.search(r"(?:no them|ghi no them).*?(?:tra dc|tra duoc)", q_no_acc)
        ):
            return "list"

        # b. Hỏi danh sách đối tượng: những ai, top N, các mặt hàng nào, liệt kê, N khách mua nhiều nhất, cho tôi N ...
        # Mệnh đề phạm vi "Trong các món, món nào ... nhất" hỏi MỘT đối tượng -> bỏ trước khi dò dấu hiệu danh sách
        q_scope = re.sub(r"\b(?:trong|trong số|giữa)\s+(?:các|những|số)\b[^,?]*,", "", q)
        if (
            re.search(
                r"\b(những .* nào|nhung .* nao|các .* nào|cac .* nao|liệt kê|liet ke|top\s*\d+|"
                r"(?:cho\s*(?:tôi|toi|mình|minh|em|xin)\s+)?(?<!\d)(?:[2-9]|10)\s+(?:mặt hàng|mat hang|khách hàng|khach hang|khách|khach|món|mon|sản phẩm|san pham|thợ|tho|nhân viên|nhan vien|thuốc|thuoc)\b|"
                r"bao nhiêu mỗi tháng|mỗi tháng là bao nhiêu|mấy thùng mấy|bao nhiêu thùng bao nhiêu|"
                r"(?:hàng|thuốc|mặt hàng|sản phẩm)\s+nào\s+hết\s+hạn)\b",
                q_scope,
            )
            or bool(re.search(r"\b(?<!\d)(?:[2-9]|10)\s+(?:khách|khach|mặt hàng|mat hang|món|mon|sản phẩm|san pham|thợ|tho|nhân viên|nhan vien|thuốc|thuoc)\b", q_scope))
        ):
            # Lưu ý: nếu hỏi số lượng mặt hàng ế ("mấy mặt hàng ế") thì là NUMBER
            if not re.search(r"\b(?:mấy|bao nhiêu)\s+(?:mặt hàng|sản phẩm|món|loại)\s+(?:ế|bán chậm|không ai mua)\b", q):
                return "list"

        # Đếm số lượng mặt hàng ế / thuốc / khách ("mấy mặt hàng ế", "có bao nhiêu mặt hàng không bán được") -> Luôn là NUMBER
        if re.search(r"\b(?:mấy|bao nhiêu)\s+(?:mặt hàng|sản phẩm|món|loại)\s+(?:ế|bán chậm|không ai mua)\b", q):
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

    # Chủ đề "dữ liệu không có thông tin này" nằm trong assets/lexicon.json (unanswerable_topics): mỗi chủ đề gồm
    # keywords (câu hỏi hỏi về thông tin này), data_keys (tên cột/bảng cho thấy shop CÓ ghi thông tin đó), reason.
    # Từ khoá mà rulebook đang dùng có định nghĩa bị loại tự động tại chỗ dùng.
    @classmethod
    def unanswerable_topics(cls) -> List[Dict[str, Any]]:
        return load_lexicon().unanswerable_topics

    @staticmethod
    def detect_unanswerable_deterministic(
        question_obj: Dict[str, Any],
        store: Any,
        date_resolver: Any,
        rulebook_text: str = "",
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
        q_na = strip_accents(q)
        is_expiry_q = any(k in q_na for k in ["han dung", "han su dung", "hsd", "het han", "can date", "het date", "qua han"])
        if not is_expiry_q:
            if hasattr(store, "get_shop_date_range"):
                s_min, s_max = store.get_shop_date_range(bid)
            elif hasattr(store, "get_date_range"):
                s_min, s_max = store.get_date_range(bid)
            else:
                s_min, s_max = None, None

            if s_min is None and s_max is None:
                return True, "Shop không có dữ liệu bán hàng theo ngày."

            s_max_effective = max(s_max, as_of) if (as_of and s_max) else (s_max or as_of)
            # README của shop ghi rõ khoảng dữ liệu phát hành -> dùng làm khoảng chuẩn (Mục 7)
            readme_rng = RuleEngine.extract_readme_date_range(store.get_readme(bid)) if hasattr(store, "get_readme") else None
            if readme_rng:
                s_min = readme_rng[0] or s_min
                s_max_effective = readme_rng[1] or s_max_effective
            d_ctx = date_resolver.resolve(question_obj.get("question", ""), as_of)
            for p in d_ctx.get("detected_periods", []):
                st = p.get("start_date")
                en = p.get("end_date")
                if st and en:
                    if en < s_min or (s_max_effective and st > s_max_effective):
                        return True, f"Kỳ hỏi ({st} đến {en}) nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop ({s_min} đến {s_max_effective})."
                    if as_of and st > as_of and any(k in q for k in ["doanh thu", "bán được", "đơn hàng", "tiền hàng", "bán chạy", "khách mua"]):
                        return True, f"Kỳ hỏi ({st} đến {en}) nằm trong tương lai sau ngày hiện tại ({as_of}), chưa có dữ liệu giao dịch phát sinh."

            # Câu so sánh với kỳ liền trước / cùng kỳ năm trước: kỳ so sánh cũng phải có dữ liệu (nếu nằm hoàn toàn ngoài
            # khoảng ngày của shop thì không đủ dữ liệu để so, Mục 7)
            explicit = [p for p in d_ctx.get("detected_periods", []) if p.get("year_month")]
            if len(explicit) == 1 and s_min:
                ym_y, ym_m = int(explicit[0]["year_month"][:4]), int(explicit[0]["year_month"][5:7])
                base = None
                if re.search(r"so voi thang (?:lien )?(?:truoc|trc)|thang lien (?:truoc|trc)|so thang (?:truoc|trc)", q_na):
                    base = (ym_y - 1, 12) if ym_m == 1 else (ym_y, ym_m - 1)
                elif re.search(r"cung ky (?:cua )?nam (?:ngoai|truoc|trc)", q_na):
                    base = (ym_y - 1, ym_m)
                if base is not None:
                    import calendar as _cal
                    b_end = f"{base[0]:04d}-{base[1]:02d}-{_cal.monthrange(base[0], base[1])[1]:02d}"
                    if b_end < s_min:
                        return True, f"Kỳ so sánh ({base[0]:04d}-{base[1]:02d}) nằm hoàn toàn trước khoảng ngày có dữ liệu của shop ({s_min})."

        tables = store.load_tables(bid) if hasattr(store, "load_tables") else {}
        all_cols = {c.lower() for df in tables.values() for c in getattr(df, "columns", [])}
        all_tbl_names = {t.lower() for t in tables.keys()}

        # 2. Câu hỏi cảm tính / thái độ / kỹ năng / chất lượng dịch vụ
        def _has_col(keys: List[str]) -> bool:
            return any(any(k in c for k in keys) for c in all_cols) or any(any(k in t for k in keys) for t in all_tbl_names)

        def _kw(keys: List[str]) -> bool:
            """Từ khoá có trong câu hỏi (theo ranh giới từ; cụm nhiều từ so cả dạng không dấu)."""
            return any(keyword_in_question(k, q, q_na) for k in keys)

        cat_values = store.get_category_values(bid) if hasattr(store, "get_category_values") else set()

        def _is_data_value(keys: List[str]) -> bool:
            """Từ khoá thực ra là TÊN nhóm/giá trị trong dữ liệu của shop (vd nhóm hàng "Dị ứng") -> không phải câu hỏi y khoa."""
            hit = [k for k in keys if keyword_in_question(k, q, q_na)]
            return bool(hit) and all(any(strip_accents(k) in v for v in cat_values) for k in hit)


        # Ngân hàng chủ đề "dữ liệu không có thông tin này": mỗi chủ đề = (từ khoá trong câu hỏi) VÀ (shop không có cột/bảng
        # tương ứng) VÀ (từ khoá không phải giá trị trong dữ liệu) VÀ (từ khoá không phải thuật ngữ được rulebook định nghĩa).
        rb_low = unicodedata.normalize("NFC", rulebook_text).lower() if rulebook_text else ""
        rb_na = strip_accents(rb_low)
        for topic in RuleEngine.unanswerable_topics():
            hits = [k for k in topic["keywords"] if keyword_in_question(k, q, q_na)]
            hits = [k for k in hits if not (rb_low and (k in rb_low or strip_accents(k) in rb_na))]
            if not hits or _has_col(topic["data_keys"]) or _is_data_value(hits):
                continue
            return True, topic["reason"]

        # 5. Địa chỉ / đường phố của khách hàng (nhận diện theo cấu trúc ngữ cảnh địa chỉ, không hardcode từ điển)
        is_street_address_query = bool(re.search(
            r"\b(?:khach\s+(?:hang\s+)?(?:nha\s+)?(?:o|tai|song)\s+(?:duong|quan|phuong|huyen|tinh|thanh pho|tp)|khach\s+(?:hang\s+)?(?:duong|quan|phuong)\s+\w|"
            r"(?:tai|o|nha o)\s+(?:duong|quan|phuong|huyen)\s+\w|dia\s+chi\s+(?:nha|cua\s+khach|giao)|ten\s+duong\s+cua\s+khach)\b",
            q_na
        ))
        if is_street_address_query:
            if not any(any(k in c for k in ["địa chỉ", "dia_chi", "đường", "duong", "address"]) for c in all_cols):
                return True, "Shop không có thông tin địa chỉ hoặc tên đường của khách hàng."

        # 6. Giới tính khách hàng
        # "khách nam" có thể là tên người ("khách Nam"): chỉ coi là giới tính khi "Nam" không phải tên khách trong dữ liệu
        if _kw(["giới tính", "nam hay nữ", "nữ hay nam", "nam giới", "nữ giới", "khách nữ", "khách hàng nữ", "phái nữ", "khách nam", "khách hàng nam"]):
            if not any(any(k in c for k in ["giới tính", "gioi_tinh", "gender"]) for c in all_cols):
                # "khách Nam" / "khách Nữ" viết hoa là tên người; viết thường "khách nam" là giới tính
                raw_q = unicodedata.normalize("NFC", question_obj.get("question", ""))
                gender_word_is_name = bool(re.search(r"[Kk]hách (?:hàng )?(?:Nam|Nữ)\b", raw_q))
                if not gender_word_is_name:
                    return True, "Shop không có cột ghi nhận giới tính của khách hàng."

        # 7. Lãi gộp / lãi ròng khi shop không có cột giá vốn
        # Chỉ khi rulebook đang dùng cấm tự tính lại giá vốn (giá vốn phải lấy trên dòng bán)
        cogs_from_line_only = not rulebook_text or bool(re.search(r"không tự tính lại giá vốn", rulebook_text, re.I))
        if cogs_from_line_only and any(k in q for k in ["lãi gộp", "lai gop", "lợi nhuận gộp", "lãi ròng", "lai rong"]):
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
            has_stock_tbl = any(any(k in t for k in ["ton_kho", "tonkho", "tồn kho", "tồn_kho", "nhap", "nhập", "kho", "stock", "inventory"]) for t in all_tbl_names)
            # "Tồn nhỏ nhất/lớn nhất", "định mức tồn" là ngưỡng cấu hình, KHÔNG phải số tồn thực tế
            _thresholds = ("nhỏ nhất", "lớn nhất", "tối thiểu", "tối đa", "định mức")
            has_stock_col = any(
                any(k in c for k in ["tồn", "ton_", "ton kho", "sl nhập", "số lượng nhập", "nhập", "stock", "inventory"])
                and not any(t in c for t in _thresholds)
                for c in all_cols
            )
            if not (has_stock_tbl or has_stock_col):
                return True, "Shop chỉ có dữ liệu bán hàng, không có bảng tồn kho hoặc dữ liệu nhập hàng để tính tồn kho."

        return False, ""

    # Chủ đề câu hỏi -> từ khoá nhận diện trong câu hỏi và từ khoá nhận diện TIÊU ĐỀ mục rulebook
    TOPICS = (
        ("ranking", ("bán chạy", "chạy nhất", "nhiều nhất", "hot nhất", "lời nhất", "xếp hạng", "top", "nhân viên", "trung bình",
                     "bình quân", "nhóm hàng", "khách mua", "khách quen", "liệt kê", "bán được bao nhiêu", "bán được mấy", "mặt hàng"),
         ("xếp hạng",), 68),
        ("debt", ("nợ", "công nợ", "trả nợ", "thu nợ", "chưa trả"), ("nợ",), 80),
        ("stock", ("nhập", "tồn", "kho", "giá vốn", "lãi", "lời", "hết hàng", "ế", "bảng giá", "tăng giá", "đơn giá", "giá bán",
                   "thùng", "hộp", "lốc", "hạn dùng", "hết hạn", "hạn sử dụng", "lô ", "chi phí"),
         ("kho", "giá vốn", "nhập"), 80),
    )

    def select_relevant_rules(self, question: str, biz_readme: str = "") -> str:
        """
        Trích các mục rulebook liên quan, nhận diện mục theo TỪ KHOÁ TIÊU ĐỀ (không theo số mục cố định của một
        rulebook cụ thể) và đưa nguyên văn vào prompt. Không thêm luật nào ngoài rulebook.
        """
        q = question.lower()
        readme_lower = biz_readme.lower()
        scored_blocks: List[Tuple[int, str]] = []
        used: set = set()

        def _add(title: Optional[str], prio: int) -> None:
            if title and title not in used and self.sections.get(title):
                used.add(title)
                scored_blocks.append((prio, self.sections[title]))

        _add("intro", 95)
        _add(self._section_title("đơn hàng", "doanh thu"), 70)
        _add(self._section_title("thời gian"), 72)
        for _, q_keys, title_keys, prio in self.TOPICS:
            if any(k in q for k in q_keys):
                _add(self._section_title(*title_keys), prio)

        # Luật ngành: mục con của mục "theo ngành", chọn khi tên ngành hoặc thuật ngữ in đậm của nó xuất hiện
        for title, body in self.sections.items():
            if " - " not in title or "ngành" not in title.lower():
                continue
            ind_name = title.split(" - ", 1)[1].strip().lower()
            bold_terms = [t.lower().strip() for t in re.findall(r"\*\*([^\*]+)\*\*", body) if len(t.strip()) >= 3]
            if (ind_name in q or ind_name in readme_lower or strip_accents(ind_name) in strip_accents(biz_readme)
                    or any(t in q or t in readme_lower for t in bold_terms)):
                _add(title, 90)

        ans_title = self._section_title("đáp án", "chấm điểm")
        if ans_title:
            bullets = [ln for ln in self.sections[ans_title].splitlines() if ln.startswith("- **")]
            if bullets:
                used.add(ans_title)
                scored_blocks.append((77, f"## {ans_title} (trích)\n" + "\n".join(bullets)))
        _add(self._section_title("không trả lời"), 60)
        _add(self._section_title("từ ngữ"), 55)
        _add(self._section_title("cách ghi"), 30)

        budget = 8000
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        kept_blocks: List[str] = []
        c_len = 0
        for _, body in scored_blocks:
            if c_len + len(body) <= budget or not kept_blocks:
                kept_blocks.append(body)
                c_len += len(body)
        return "\n\n".join(kept_blocks)

    @staticmethod
    def extract_readme_date_range(biz_readme: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Đọc khoảng ngày dữ liệu README công bố: 'Dữ liệu ... từ DD/MM/YYYY đến (hết) DD/MM/YYYY'
        hoặc 'Dữ liệu có đến hết DD/MM/YYYY'. Trả về (start_iso, end_iso) hoặc None.
        """
        if not biz_readme:
            return None
        text = unicodedata.normalize("NFC", biz_readme)
        d_pat = r"(\d{1,2})[/-](\d{1,2})[/-](20\d{2})"

        def _iso(m: Any, off: int) -> Optional[str]:
            try:
                return date(int(m.group(off + 3)), int(m.group(off + 2)), int(m.group(off + 1))).isoformat()
            except ValueError:
                return None

        m_rng = re.search(r"[Dd]ữ liệu[^\n.]*?từ\s*(?:ngày\s*)?" + d_pat + r"\s*(?:đến|tới|-|–)\s*(?:hết\s*)?(?:ngày\s*)?" + d_pat, text)
        if m_rng:
            return _iso(m_rng, 0), _iso(m_rng, 3)
        m_end = re.search(r"[Dd]ữ liệu[^\n.]*?đến hết\s*(?:ngày\s*)?" + d_pat, text)
        if m_end:
            return None, _iso(m_end, 0)
        return None

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
    Few-shot theo cách TỔ CHỨC ví dụ của DAIL-SQL (DAIL-O: chỉ giữ cặp câu hỏi -> SQL, bỏ schema của DB ví dụ).
    Phần CHỌN ví dụ là luật từ khoá theo dạng câu hỏi, KHÔNG phải DAIL-S của bài báo.
    Các ví dụ chỉ minh hoạ CÚ PHÁP DuckDB; điều kiện nghiệp vụ để dạng <...theo rulebook> để mô hình điền từ
    rulebook đang dùng (không nhúng luật của một rulebook cụ thể). Không có chú thích trong SQL mẫu để mô hình
    không bắt chước viết chú thích.
    """

    VALID = "<điều kiện bản ghi được tính theo rulebook>"

    EXEMPLARS = {
        "period_sum": (
            "Câu hỏi: Tổng <đại lượng> từ ngày a đến ngày b là bao nhiêu?\n"
            "SQL:\n"
            "SELECT COALESCE(SUM(\"<cột đại lượng>\"), 0)\n"
            "FROM <bảng>\n"
            "WHERE {valid}\n"
            "  AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b';"
        ),
        "growth": (
            "Câu hỏi: <Đại lượng> kỳ A thay đổi bao nhiêu % so với kỳ B?\n"
            "SQL:\n"
            "WITH ky AS (\n"
            "  SELECT\n"
            "    SUM(CASE WHEN CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a1' AND DATE 'a2' THEN \"<cột đại lượng>\" ELSE 0 END) AS ky_a,\n"
            "    SUM(CASE WHEN CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'b1' AND DATE 'b2' THEN \"<cột đại lượng>\" ELSE 0 END) AS ky_b\n"
            "  FROM <bảng> WHERE {valid}\n"
            ")\n"
            "SELECT (ky_a - ky_b) * 100.0 / NULLIF(ky_b, 0) FROM ky;"
        ),
        "top_k": (
            "Câu hỏi: k <đối tượng> có <tiêu chí> cao nhất kỳ a-b là gì?\n"
            "SQL:\n"
            "SELECT ANY_VALUE(\"<cột tên>\") AS ten\n"
            "FROM <bảng>\n"
            "WHERE {valid} AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b'\n"
            "GROUP BY \"<cột mã>\"\n"
            "ORDER BY SUM(\"<cột tiêu chí>\") DESC, \"<cột mã>\" ASC\n"
            "LIMIT k;"
        ),
        "count_distinct": (
            "Câu hỏi: Có bao nhiêu <đơn / khách> trong kỳ a-b?\n"
            "SQL:\n"
            "SELECT COUNT(DISTINCT \"<cột mã>\")\n"
            "FROM <bảng>\n"
            "WHERE {valid} AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b';"
        ),
        "share": (
            "Câu hỏi: <Nhóm X> chiếm bao nhiêu % <đại lượng> kỳ a-b?\n"
            "SQL:\n"
            "SELECT SUM(CASE WHEN \"<cột nhóm>\" = '<X>' THEN \"<cột đại lượng>\" ELSE 0 END) * 100.0\n"
            "       / NULLIF(SUM(\"<cột đại lượng>\"), 0)\n"
            "FROM <bảng>\n"
            "WHERE {valid} AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b';"
        ),
        "ratio_of_records": (
            "Câu hỏi: Tỉ lệ <loại bản ghi Y> trong kỳ a-b là bao nhiêu %?\n"
            "SQL:\n"
            "SELECT COUNT(DISTINCT CASE WHEN <điều kiện Y> THEN \"<cột mã>\" END) * 100.0\n"
            "       / NULLIF(COUNT(DISTINCT \"<cột mã>\"), 0)\n"
            "FROM <bảng>\n"
            "WHERE <điều kiện mẫu số theo rulebook> AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b';"
        ),
        "hour_bucket": (
            "Câu hỏi: So <đại lượng> giữa hai khung giờ h1-h2 và h3-h4?\n"
            "SQL:\n"
            "SELECT\n"
            "  SUM(CASE WHEN hour(CAST(\"<cột thời gian>\" AS TIMESTAMP)) BETWEEN h1 AND h2 THEN \"<cột đại lượng>\" ELSE 0 END),\n"
            "  SUM(CASE WHEN hour(CAST(\"<cột thời gian>\" AS TIMESTAMP)) BETWEEN h3 AND h4 THEN \"<cột đại lượng>\" ELSE 0 END)\n"
            "FROM <bảng> WHERE {valid};"
        ),
        "entity_filter": (
            "Câu hỏi: <Đại lượng> của mặt hàng 'tên gõ không dấu' kỳ a-b?\n"
            "SQL:\n"
            "SELECT COALESCE(SUM(\"<cột đại lượng>\"), 0)\n"
            "FROM <bảng>\n"
            "WHERE {valid} AND CAST(\"<cột ngày>\" AS DATE) BETWEEN DATE 'a' AND DATE 'b'\n"
            "  AND strip_accents(\"<cột tên>\") ILIKE strip_accents('%tên gõ không dấu%');"
        ),
        "trend_boolean": (
            "Câu hỏi: <Đại lượng> các tháng m1, m2, m3 có tăng liên tục không?\n"
            "SQL:\n"
            "WITH theo_thang AS (\n"
            "  SELECT SUBSTR(\"<cột ngày>\", 1, 7) AS ym, SUM(\"<cột đại lượng>\") AS v\n"
            "  FROM <bảng> WHERE {valid} GROUP BY ym\n"
            ")\n"
            "SELECT (SELECT v FROM theo_thang WHERE ym = 'm1') < (SELECT v FROM theo_thang WHERE ym = 'm2')\n"
            "   AND (SELECT v FROM theo_thang WHERE ym = 'm2') < (SELECT v FROM theo_thang WHERE ym = 'm3');"
        ),
    }

    INTENTS = (
        ("ratio_of_records", ("ti le", "ty le")),
        ("growth", ("so voi", "tang bao nhieu %", "giam bao nhieu %", "thay doi", "tang truong", "cung ky")),
        ("share", ("chiem", "ti trong", "ty trong", "dong gop")),
        ("top_k", ("nhat", "top ", "xep hang")),
        ("count_distinct", ("bao nhieu don", "may don", "so don", "bao nhieu khach", "may khach", "so khach")),
        ("hour_bucket", ("ca sang", "ca chieu", "khung gio", " gio ")),
        ("trend_boolean", ("lien tuc", "luon tang", "luon giam", "tang deu")),
    )

    @classmethod
    def select_exemplars(cls, question: str, has_entity_hint: bool = False) -> str:
        """Chọn tối đa 2 ví dụ cú pháp theo dạng câu hỏi (DAIL-O: chỉ cặp câu hỏi -> SQL)."""
        q = " " + strip_accents(question).lower() + " "
        selected: List[str] = []
        for name, keys in cls.INTENTS:
            if any(k in q for k in keys) and name not in selected:
                selected.append(name)
        if has_entity_hint:
            selected.append("entity_filter")
        if not selected:
            selected.append("period_sum")
        chosen = [cls.EXEMPLARS[n].format(valid=cls.VALID) for n in selected[:2]]
        return (
            "=== VÍ DỤ CÚ PHÁP DUCKDB (chỉ minh hoạ cấu trúc; thay <...> bằng bảng, cột và luật thật) ===\n"
            + "\n\n".join(chosen)
            + "\n"
        )
