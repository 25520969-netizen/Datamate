# -*- coding: utf-8 -*-
"""
Module: Vietnamese Context-Aware Canonicalizer (Chuẩn hoá tiếng Việt theo ngữ cảnh DB)
Dựa trên cơ sở lý thuyết và thực nghiệm từ hai công trình học thuật:
1. ViLexNorm (EACL 2024): Thống kê và xử lý biến thể hình thái tiếng Việt mạng xã hội
   (từ ngữ địa phương, từ viết tắt, teen code, lỗi chính tả Telex).
2. DART-SQL (ACL Findings 2024 - Section 3.1): Question Rewriting leveraging Database Content
   để loại bỏ sự mơ hồ ngữ cảnh (Context Ambiguity) mà không cần hardcode từ điển tĩnh.
"""

import re
import unicodedata

from src.lexicon import load_lexicon
from typing import Any, Dict, List, Optional, Tuple


def to_nfc(text: str) -> str:
    """Chuẩn hóa chuỗi Unicode sang chuẩn NFC."""
    return unicodedata.normalize("NFC", text or "")


class VietnameseNormalizer:
    """
    Bộ chuẩn hoá câu hỏi tiếng Việt tổng quát hoá:
    - Lớp 1 (Deterministic): Chuẩn hóa Unicode NFC, mở rộng viết tắt nghiệp vụ, loại bỏ khoảng trắng thừa.
    - Lớp 2 (Context-Aware LLM Rewriting): Sử dụng động cơ Qwen2.5-Coder (chạy offline local) để chuẩn hoá
      các từ địa phương, teen code và từ viết tắt dựa trên cấu trúc bảng/cột của DB mà không cần từ điển cứng.
    """

    SYSTEM_PROMPT_CANONICALIZER = (
        "Bạn là chuyên gia ngôn ngữ tiếng Việt hỗ trợ hệ thống cơ sở dữ liệu doanh nghiệp.\n"
        "Nhiệm vụ: Viết lại câu hỏi sau về dạng tiếng Việt chuẩn tắc, rõ ràng (canonical Vietnamese).\n"
        "Quy tắc bắt buộc:\n"
        "1. Làm rõ các từ viết tắt, tiếng địa phương, teen code và sửa lỗi chính tả Telex dựa trên danh sách cột của cơ sở dữ liệu.\n"
        "2. Giữ nguyên toàn bộ nghĩa gốc, các con số, mã đơn hàng, mã sản phẩm hoặc tên đối tượng trong dấu ngoặc kép.\n"
        "3. Tuyệt đối KHÔNG thêm bớt điều kiện lọc, không tự bịa thêm thông tin ngoài câu hỏi gốc.\n"
        "4. Nếu câu hỏi đã chuẩn mực và rõ ràng, hãy giữ nguyên câu hỏi.\n"
        "Chỉ trả về DUY NHẤT một câu hỏi đã chuẩn hoá, không giải thích gì thêm."
    )

    # Viết tắt / tiếng lóng lấy từ từ điển assets/lexicon.json (học từ bộ train, có điều kiện theo ngữ cảnh)

    # Không giữ danh sách lỗi gõ cụ thể (trước đây chép từ câu hỏi bộ train: "naof", "cuwar", "bani"...):
    # lỗi gõ được xử lý chung ở bước viết lại câu hỏi bằng LLM (canonicalize).

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client
        self._cache: Dict[str, str] = {}

    @classmethod
    def mask_entities(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Bảo vệ tuyệt đối các thực thể nhạy cảm trước khi đưa vào chuẩn hoá:
        - Chuỗi trong ngoặc kép hoặc ngoặc đơn: "..." hoặc '...'
        - Hậu tố kích cỡ size F&B: (M), (L), (S), (XL), (XXL)
        - Mã định danh: HD..., SP..., KH..., NCC...
        """
        entity_map: Dict[str, str] = {}
        counter = 0

        def _repl(match: Any) -> str:
            nonlocal counter
            token = f"__PROTECTED_ENTITY_{counter}__"
            entity_map[token] = match.group(0)
            counter += 1
            return token

        # 1. Bảo vệ chuỗi trong ngoặc kép / đơn
        masked = re.sub(r'("[^"]+"|\'[^\']+\')', _repl, text)
        # 2. Bảo vệ hậu tố size trong ngoặc đơn (M, L, S, XL)
        masked = re.sub(r'\(\s*(?:[smlSML]|XL|XXL|Xl|xxl)\s*\)', _repl, masked)
        # 3. Bảo vệ mã định danh (HD123, SP456...)
        masked = re.sub(r'\b(?:HD|HĐ|SP|KH|NCC|PN|PX)\d+\b', _repl, masked, flags=re.IGNORECASE)

        return masked, entity_map

    @classmethod
    def unmask_entities(cls, text: str, entity_map: Dict[str, str]) -> str:
        """Khôi phục lại nguyên trạng các thực thể đã được bảo vệ."""
        res = text
        for token, orig in entity_map.items():
            res = res.replace(token, orig)
        return res

    @classmethod
    def clean_deterministic(cls, text: str) -> str:
        """
        Chuẩn hoá cơ bản bằng quy tắc ngôn ngữ học, Unicode NFC và bảo vệ thực thể.
        """
        if not text:
            return ""
        s = to_nfc(text.strip())
        masked, entity_map = cls.mask_entities(s)

        masked = load_lexicon().expand_query(masked)

        # Khôi phục thực thể bảo vệ
        restored = cls.unmask_entities(masked, entity_map)
        # Chuẩn hóa khoảng trắng
        return re.sub(r"\s+", " ", restored).strip()

    @staticmethod
    def extract_cell_samples(question: str, tables: Optional[Dict[str, Any]] = None, max_samples: int = 4) -> str:
        """
        Trích xuất các giá trị ô thực tế từ DB khớp với các từ trong câu hỏi (DART-SQL cell grounding).
        """
        if not tables or not isinstance(tables, dict):
            return ""
        
        # Lọc các từ có độ dài >= 2 và loại bỏ stop words phổ biến
        stop_words = {"và", "là", "có", "ở", "ra", "vô", "gì", "ai", "cho", "của", "tại", "với", "được", "các", "những", "này", "đó", "khi", "nào", "mấy", "bao", "nhiêu", "tổng", "tất", "cả"}
        raw_tokens = [w.lower() for w in re.split(r"\W+", question) if len(w) >= 2]
        q_tokens = [w for w in raw_tokens if w not in stop_words]
        if not q_tokens:
            return ""

        found_samples: List[str] = []
        seen: set = set()

        for df in tables.values():
            if not hasattr(df, "columns") or len(df) == 0:
                continue
            for col in df.columns:
                c_low = str(col).lower()
                if any(k in c_low for k in ["tên", "ten", "sản phẩm", "hàng", "khách", "mặt hàng", "mã"]):
                    sample_series = df[col].dropna().astype(str).unique()
                    for val in sample_series[:200]:
                        v_str = str(val).strip()
                        v_low = v_str.lower()
                        if not v_str or v_str in seen:
                            continue
                        
                        # Khớp theo từ đơn hoặc cụm từ
                        matched = False
                        for tok in q_tokens:
                            if re.search(r"(?:\b|^)" + re.escape(tok) + r"(?:\b|$)", v_low, re.IGNORECASE):
                                matched = True
                                break
                        
                        if matched:
                            seen.add(v_str)
                            found_samples.append(f"- '{v_str}' (cột '{col}')")
                            if len(found_samples) >= max_samples:
                                break
                if len(found_samples) >= max_samples:
                    break
            if len(found_samples) >= max_samples:
                break

        if not found_samples:
            return ""
        return "Mẫu giá trị thực tế trong cơ sở dữ liệu:\n" + "\n".join(found_samples)

    def canonicalize(
        self,
        question: str,
        schema_summary: str = "",
        as_of_date: str = "",
        tables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Chuẩn hoá câu hỏi tiếng Việt kết hợp ngữ cảnh cơ sở dữ liệu (DART-SQL) và bảo vệ thực thể.
        """
        base_clean = self.clean_deterministic(question)
        if not base_clean:
            return ""

        # Bảo vệ thực thể định danh trước khi đưa qua LLM
        masked_clean, entity_map = self.mask_entities(base_clean)

        cache_key = f"{masked_clean}___{schema_summary[:200]}"
        if cache_key in self._cache:
            return self.unmask_entities(self._cache[cache_key], entity_map)

        if self.llm_client is None:
            self._cache[cache_key] = masked_clean
            return base_clean

        backend = getattr(self.llm_client, "backend", "")
        if backend in ("mock", "custom") and not getattr(self.llm_client, "_supports_canonicalize", False):
            self._cache[cache_key] = masked_clean
            return base_clean

        cell_samples = self.extract_cell_samples(base_clean, tables=tables)

        user_content = f"Cấu trúc bảng và các cột dữ liệu hiện có:\n{schema_summary[:1500]}\n\n"
        if cell_samples:
            user_content += f"{cell_samples}\n\n"
        if as_of_date:
            user_content += f"Mốc ngày hiện tại (as_of_date): {as_of_date}\n\n"
        user_content += f"Câu hỏi gốc cần chuẩn hoá:\n\"{masked_clean}\"\n\nCâu hỏi chuẩn hoá:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT_CANONICALIZER},
            {"role": "user", "content": user_content},
        ]

        try:
            raw_reply = self.llm_client.generate(messages, seed=42)
            cleaned_reply = raw_reply.strip()
            if cleaned_reply.startswith('"') and cleaned_reply.endswith('"') and len(cleaned_reply) >= 2:
                cleaned_reply = cleaned_reply[1:-1].strip()
            if cleaned_reply.startswith("Câu hỏi chuẩn hoá:"):
                cleaned_reply = cleaned_reply[len("Câu hỏi chuẩn hoá:"):].strip()

            # Bản viết lại phải giữ nguyên mọi thực thể đã che; mất token nghĩa là LLM đã đổi nghĩa câu hỏi
            if (
                len(cleaned_reply) >= 5
                and not cleaned_reply.startswith("```")
                and all(tok in cleaned_reply for tok in entity_map)
            ):
                # Cache bản còn che thực thể: hai câu chỉ khác thực thể ("(M)" vs "(L)") dùng chung khóa cache
                self._cache[cache_key] = to_nfc(cleaned_reply)
                return self.unmask_entities(self._cache[cache_key], entity_map)
        except Exception:
            pass

        self._cache[cache_key] = masked_clean
        return base_clean
