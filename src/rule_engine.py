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

        if not matched_hints:
            return ""

        return "=== GỢI Ý THỰC THỂ KHỚP TRONG CƠ SỞ DỮ LIỆU ===\n" + "\n".join(matched_hints)


DEFAULT_RULEBOOK_PATH = Path(__file__).resolve().parent.parent / "train" / "rulebook.md"


class RuleEngine:
    """Bộ phân tích câu hỏi và trích xuất luật nghiệp vụ theo BIRD Knowledge Injection paradigm."""

    def __init__(self, rulebook_path: Path = DEFAULT_RULEBOOK_PATH) -> None:
        self.rulebook_path = rulebook_path
        self.sections: Dict[str, str] = {}
        self.full_rulebook: str = ""
        self._load_rulebook()

    def _load_rulebook(self) -> None:
        if not self.rulebook_path.exists():
            return
        self.full_rulebook = self.rulebook_path.read_text(encoding="utf-8")
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
            re.search(r"\b(?:top\s*\d+|danh\s*sách|bảng)\b.*\b(?:kèm|và)\s+(?:số\s+tiền|doanh\s+thu|số\s+lượng|số\s+nợ|tiền\s+nợ|công\s+nợ|hoa\s+hồng)", q)
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
        if re.search(
            r"\b(có phải|co phai|đúng không|dung khong|đúng hay sai|có hay không|co hay khong|"
            r"tăng liên tục|tang lien tuc|giảm liên tục|giam lien tuc|luôn cao hơn|luon cao hon|"
            r"luôn lớn hơn|luon lon hon|có lớn hơn .* không|có cao hơn .* không|có vượt .* không|"
            r"có luôn hơn .* không|co luon hon .* khong|có tăng liên tục|co tang lien tuc|"
            r"có luôn .* không|co luon .* khong|có giảm liên tục|co giam lien tuc)\b",
            q,
        ):
            return "boolean"

        # 3. Percent: phần trăm, tỉ lệ, tỷ lệ, tỉ trọng, tỷ trọng, bao nhiêu %, mấy %, tăng/giảm bao nhiêu %
        # Loại trừ % nồng độ sản phẩm trong câu hỏi (ví dụ: cồn 70%, nước muối 0,9%, otrivin 0,1%, kem 5%)
        q_no_prod_pct = re.sub(r"\b\d+[\.,]?\d*\s*%", "", q)
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
                    "tỷ số",
                    "tỉ số",
                    "tăng hay giảm bao nhiêu %",
                    "tăng hay giảm bao nhiêu phần trăm",
                    "thay đổi mấy %",
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
        # a. Hỏi 2 hoặc nhiều giá trị: "mỗi loại bao nhiêu", "lần lượt là bao nhiêu", "theo thứ tự đó", "mấy ... chẵn ... lẻ", "tuổi nợ"
        if re.search(
            r"\b(mỗi loại bao nhiêu|mỗi loại bn|mỗi bên bao nhiêu|mỗi bên bn|mỗi thứ bao nhiêu|mỗi thứ bn|"
            r"lần lượt là bao nhiêu|lần lượt là bn|lần lượt theo thứ tự|theo thứ tự đó|ghi hai số|ghi 2 số|"
            r"chẵn.*lẻ|chan.*le|hộp chẵn|vỉ chẵn|thùng chẵn|"
            r"tuổi nợ|tuoi no)\b",
            q,
        ):
            return "list"
        # b. Hỏi danh sách đối tượng: những ai, top N, các mặt hàng nào, liệt kê, N khách mua nhiều nhất, cho tôi N ...
        if re.search(
            r"\b(những .* nào|nhung .* nao|các .* nào|cac .* nao|liệt kê|liet ke|top\s*\d+|"
            r"(?:cho\s*(?:tôi|mình|em|xin)\s+)?(?<!\d)(?:[2-9]|10)\s+(?:mặt hàng|khách hàng|khách|món|sản phẩm|thợ|nhân viên|thuốc)\b|"
            r"bao nhiêu mỗi tháng|mỗi tháng là bao nhiêu|mấy thùng mấy|bao nhiêu thùng bao nhiêu)\b",
            q,
        ):
            return "list"

        # Đếm số lượng mặt hàng ế / thuốc / khách ("mấy mặt hàng ế", "có bao nhiêu mặt hàng không bán được") -> Luôn là NUMBER
        if re.search(r"\b(?:mấy|bao nhiêu)\s+(?:mặt hàng|sản phẩm|món|loại)\s+(?:ế|ees|bán chậm|không ai mua)\b", q):
            return "number"

        # 6. String: hỏi đích danh 1 đối tượng duy nhất (ai, hàng nào, sản phẩm nào, món nào, khách nào, ncc nào, là gì)
        if re.search(
            r"\b(ai là|ai la|ai đang|ai dang|ai trung thành|ai sộp|ai hay|ai được|ai bán|ai mua|ai có|ai chi|là ai|la ai|"
            r"mặt hàng nào|mat hang nao|hàng nào|hang nao|sản phẩm nào|san pham nao|món nào|mon nao|"
            r"khách hàng nào|khach hang nao|khách nào|khach nao|nhà cung cấp nào|ncc nào|nhân viên nào|thợ nào|tho nao|"
            r"lô nào|lo nao|nhóm nào|kênh nào|hình thức nào|chi nhánh nào|thứ mấy|thu may|hàng hot nhất|mặt hàng hot nhất|"
            r"là gì|la gi)\b",
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
            d_ctx = date_resolver.resolve(question_obj.get("question", ""), as_of)
            for p in d_ctx.get("detected_periods", []):
                st = p.get("start_date")
                en = p.get("end_date")
                if st and en:
                    if en < s_min or st > s_max:
                        return True, f"Kỳ hỏi ({st} đến {en}) nằm hoàn toàn ngoài khoảng ngày có dữ liệu của shop ({s_min} đến {s_max})."

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
            has_cogs = any("giá vốn" in c or "gia_von" in c for c in all_cols)
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

        # Mục 3: Xếp hạng / khách hàng / bán chạy / trung bình đơn
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
                "khách",
                "nhân viên",
                "trung bình",
                "bình quân",
                "nhóm hàng",
                "phí phục vụ",
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
        ) or bool(re.search(r"\b(?:lời|tiền lời|sinh lời|có lời|hàng ế|tồn ế)\b", q))
        if has_sec5:
            selected_keys.append("5. Kho, nhập hàng, giá vốn")

        # Mục 6: Theo ngành (chỉ nạp tiểu mục của đúng ngành cần thiết, tiết kiệm ~4000 ký tự prompt)
        if any(kw in q or kw in readme_lower for kw in ["sửa xe", "sua xe", "thợ", "nhớt", "lượt xe"]):
            selected_keys.append("6. Theo ngành - Sửa xe")
        elif any(kw in q or kw in readme_lower for kw in ["điện thoại", "dien thoai", "máy cũ", "trả góp", "dòng máy", "imei", "bảo hành"]):
            selected_keys.append("6. Theo ngành - Cửa hàng điện thoại")
        elif any(kw in q or kw in readme_lower for kw in ["bỉm", "tã", "mẹ và bé", "thành viên mới", "voucher", "dùng điểm"]):
            selected_keys.append("6. Theo ngành - Mẹ và bé")
        elif any(kw in q or kw in readme_lower for kw in ["thuốc", "kê đơn", "hoạt chất", "hạn dùng", "hạn sử dụng", "lô"]):
            selected_keys.append("6. Theo ngành - Nhà thuốc")
        elif any(kw in q or kw in readme_lower for kw in ["cà phê", "cafe", "size ly", "topping"]):
            selected_keys.append("6. Theo ngành - Cà phê")
        elif any(kw in q or kw in readme_lower for kw in ["xi măng", "gạch", "vật liệu", "đợt giao", "chở giao"]):
            selected_keys.append("6. Theo ngành - Vật liệu xây dựng")

        # Tóm tắt cô đọng Mục 2 (Thời gian & Ranh giới) để tiết kiệm context window
        compact_footer = (
            "## Quy tắc Thời gian & Ranh giới (Mục 2):\n"
            "- 'Quá N, hơn N, trên N, dưới N': không gồm N; 'Từ N trở lên, từ N trở xuống': gồm N.\n"
            "- Ca sáng: 06:00–13:59, ca chiều: 14:00–21:59 (so ca chiều với ca sáng: mốc là ca sáng).\n"
            "- Hỏi số còn lại ('tính đến hết ngày X', 'còn nợ', 'còn tồn'): tính mọi phát sinh từ đầu dữ liệu đến hết ngày đó."
        )

        seen = set()
        blocks: List[str] = []
        for k in selected_keys:
            if k in seen:
                continue
            seen.add(k)
            for sec_title, sec_body in self.sections.items():
                if sec_title == k or sec_title.startswith(k):
                    blocks.append(sec_body)
                    break

        blocks.append(compact_footer)
        return "\n\n".join(blocks)

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
