# -*- coding: utf-8 -*-
"""
Từ điển ngôn ngữ - dữ liệu (assets/lexicon.json) và các hàm dùng nó trong pipeline.

Từ điển được học/kiểm chứng CHỈ từ bộ train (xem tools/mine_lexicon.py), gồm:
  - query_abbreviations: viết tắt/tiếng lóng trong câu hỏi -> dạng chuẩn (có điều kiện theo từ đứng trước/sau);
  - value_concepts: cách nói tiếng Việt <-> giá trị mã hoá trong dữ liệu ("tiền mặt" <-> "cash", "tại quầy" <-> "store");
  - status_variants: biến thể trạng thái đơn -> 4 trạng thái chuẩn;
  - column_roles: tên cột theo vai trò (mở rộng bảng đồng nghĩa khi nạp dữ liệu);
  - unanswerable_topics: chủ đề "dữ liệu không có thông tin này".
"""

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "lexicon.json"


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip(s: str) -> str:
    s = _nfc(s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


class Lexicon:
    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.data: Dict[str, Any] = data or {}
        qa = self.data.get("query_abbreviations", {})
        self.always: Dict[str, str] = {_nfc(k).lower(): v for k, v in qa.get("always", {}).items()}
        self.contextual: List[Dict[str, Any]] = qa.get("contextual", [])
        self.protect_digit: Set[str] = {t.lower() for t in qa.get("protect_when_followed_by_digit", [])}
        self.value_concepts: Dict[str, Dict[str, List[str]]] = self.data.get("value_concepts", {})
        self.status_variants: Dict[str, List[str]] = self.data.get("status_variants", {})
        self.column_roles: Dict[str, List[str]] = self.data.get("column_roles", {})
        self.unanswerable_topics: List[Dict[str, Any]] = self.data.get("unanswerable_topics", [])
        self._status_lookup: Dict[str, str] = {
            _nfc(v).lower().strip(): canon for canon, vs in self.status_variants.items() for v in vs
        }
        # Cụm nhiều từ trước, từ đơn sau (tránh "bao nhiu" bị tách)
        self._always_sorted = sorted(self.always.items(), key=lambda kv: -len(kv[0]))
        self._always_re = None
        if self._always_sorted:
            alt = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k, _ in self._always_sorted)
            self._always_re = re.compile(r"(?<![\w])(" + alt + r")(?![\w])", re.IGNORECASE)
        self._cat_index: Dict[int, Any] = {}
        self._compile_rules()

    def _compile_rules(self) -> None:
        self._ctx_compiled = []
        for rule in self.contextual:
            tok = re.escape(rule["token"])
            prev, nxt = rule.get("prev"), rule.get("next")
            pat_prev = pat_next = None
            if prev:
                pv = "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in prev)
                pat_prev = re.compile(r"(?<![\w])((?:" + pv + r")\s+)" + tok + r"(?![\w])", re.IGNORECASE)
            if nxt:
                if nxt == ["%"]:
                    pat_next = re.compile(r"(?<![\w])" + tok + r"(?=\s*\d+(?:[.,]\d+)?\s*%)", re.IGNORECASE)
                else:
                    nx = "|".join(re.escape(n).replace(r"\ ", r"\s+") for n in nxt)
                    pat_next = re.compile(r"(?<![\w])" + tok + r"(?=\s+(?:" + nx + r")(?![\w]))", re.IGNORECASE)
            self._ctx_compiled.append((rule["to"], pat_prev, pat_next))
        self._concept_patterns = {}
        for name, spec in self.value_concepts.items():
            pats = []
            for ph in spec.get("phrases", []):
                p = _nfc(ph).lower()
                exact = re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)")
                na = re.compile(r"(?<!\w)" + re.escape(_strip(p)) + r"(?!\w)") if len(p.split()) >= 2 else None
                pats.append((exact, na))
            self._concept_patterns[name] = pats

    # ------------------------------------------------------------------ câu hỏi
    def expand_query(self, text: str) -> str:
        """Mở rộng viết tắt/tiếng lóng. Không đụng vào chuỗi trong ngoặc kép."""
        if not text:
            return text
        parts = re.split(r'("[^"]*"|\'[^\']*\')', text)
        for i in range(0, len(parts), 2):
            parts[i] = self._expand_segment(parts[i])
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def _expand_segment(self, seg: str) -> str:
        # 1. Điều kiện theo từ đứng trước/sau (trước khi mở rộng "luôn đúng" để còn nhận ra ngữ cảnh gốc)
        for to, pat_prev, pat_next in self._ctx_compiled:
            if pat_prev is not None:
                seg = pat_prev.sub(lambda m, to=to: m.group(1) + to, seg)
            if pat_next is not None:
                seg = pat_next.sub(to, seg)
        # 2. Luôn mở rộng (trừ khi theo sau là số: "dc 2" là mã)
        if self._always_re is not None:
            def _sub(m: "re.Match[str]") -> str:
                key = re.sub(r"\s+", " ", m.group(1).lower())
                to = self.always.get(key)
                if to is None:
                    return m.group(0)
                if key in self.protect_digit and re.match(r"\s*\d", seg[m.end():]):
                    return m.group(0)
                return to
            seg = self._always_re.sub(_sub, seg)
        return seg

    # ------------------------------------------------------------------ dữ liệu
    def canonical_status(self, raw: str) -> Optional[str]:
        """Trạng thái chuẩn theo bảng biến thể (khớp đúng toàn chuỗi), hoặc None."""
        return self._status_lookup.get(_nfc(raw).lower().strip())

    def role_names(self, role: str) -> List[str]:
        return list(self.column_roles.get(role, []))

    def concepts_in_question(self, question: str) -> List[str]:
        q = _nfc(question).lower()
        q_na = _strip(question)
        found = []
        for name, pats in self._concept_patterns.items():
            if any(exact.search(q) or (na is not None and na.search(q_na)) for exact, na in pats):
                found.append(name)
        return found

    def _categorical_index(self, tables: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, str]]]:
        """Chỉ mục cột phân loại của shop: [(bảng, cột, {giá trị bỏ dấu -> giá trị gốc})], tính một lần mỗi shop."""
        cached = self._cat_index.get(id(tables))
        if cached is not None and cached[0] is tables:
            return cached[1]
        index: List[Tuple[str, str, Dict[str, str]]] = []
        for tname, df in tables.items():
            for col in getattr(df, "columns", []):
                if df[col].dtype != object:
                    continue
                uniq = df[col].dropna().unique()
                if not (0 < len(uniq) <= 300):
                    continue
                index.append((tname, col, {_strip(u): u for u in uniq if isinstance(u, str)}))
        self._cat_index[id(tables)] = (tables, index)
        return index

    def build_value_hints(self, question: str, tables: Dict[str, Any], max_hints: int = 4) -> str:
        """
        Nối cách nói trong câu hỏi với GIÁ TRỊ THẬT trong dữ liệu của shop: chỉ gợi ý khi tìm thấy giá trị biến thể
        trong một cột chữ ít giá trị (cột phân loại), kèm bảng/cột để mô hình lọc đúng chính tả.
        """
        concepts = self.concepts_in_question(question)
        if not concepts:
            return ""
        index = self._categorical_index(tables)
        hints: List[str] = []
        for name in concepts:
            wanted = {_strip(v) for v in self.value_concepts[name].get("values", [])}
            for tname, col, vals in index:
                hit = [orig for na, orig in vals.items() if na in wanted]
                if hit:
                    hints.append(
                        f"- \"{self.value_concepts[name]['phrases'][0]}\" trong câu hỏi ứng với giá trị "
                        f"{', '.join(repr(h) for h in hit[:3])} của cột \"{col}\" (bảng \"{tname}\")"
                    )
                    break
            if len(hints) >= max_hints:
                break
        if not hints:
            return ""
        return "=== GIÁ TRỊ MÃ HOÁ TRONG DỮ LIỆU ỨNG VỚI CÁCH NÓI TRONG CÂU HỎI ===\n" + "\n".join(hints)


@lru_cache(maxsize=4)
def load_lexicon(path: Optional[str] = None) -> Lexicon:
    p = Path(path) if path else DEFAULT_LEXICON_PATH
    if not p.exists():
        return Lexicon({})
    return Lexicon(json.loads(p.read_text(encoding="utf-8")))
