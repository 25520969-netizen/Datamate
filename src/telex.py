# -*- coding: utf-8 -*-
"""
Giải mã chữ gõ Telex thô ("naof" -> "nào", "thangs" -> "tháng", "tieenf" -> "tiền").

Đây là bộ giải mã theo LUẬT gõ Telex tiếng Việt (không có danh sách từ lỗi cụ thể). Một từ chỉ được đổi khi:
  1. toàn từ là chữ thường a-z và có dấu hiệu Telex (nguyên âm đôi aa/ee/oo, aw/ow/uw, dd, hoặc phím dấu s/f/r/x/j
     đứng sau nguyên âm ở cuối âm tiết);
  2. kết quả giải mã là một âm tiết tiếng Việt hợp lệ (phụ âm đầu + vần + phụ âm cuối + quy tắc dấu với âm tắc);
  3. từ gốc không nằm trong tập từ được bảo vệ (tên mặt hàng, tên cột... của chính shop) và không viết hoa giữa câu.
"""

import re
from typing import Iterable, Optional, Set

TONE_KEYS = "sfrxj"
_TONE_INDEX = {"s": 1, "f": 2, "r": 3, "x": 4, "j": 5}  # sắc, huyền, hỏi, ngã, nặng

_VOWEL_BASE = "aăâeêioôơuưy"
_TONED = {
    "a": "aáàảãạ", "ă": "ăắằẳẵặ", "â": "âấầẩẫậ", "e": "eéèẻẽẹ", "ê": "êếềểễệ", "i": "iíìỉĩị",
    "o": "oóòỏõọ", "ô": "ôốồổỗộ", "ơ": "ơớờởỡợ", "u": "uúùủũụ", "ư": "ưứừửữự", "y": "yýỳỷỹỵ",
}

ONSETS = ("ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr",
          "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x")
CODAS = ("ch", "ng", "nh", "c", "m", "n", "p", "t")
STOP_CODAS = {"c", "ch", "p", "t"}  # chỉ đi với dấu sắc hoặc nặng

NUCLEI = {
    "a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y",
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu", "oa", "oă", "oe", "oi", "ôi", "ơi",
    "ua", "uâ", "uê", "ui", "uô", "uy", "ưa", "ưi", "ươ", "ưu", "yê",
    "iêu", "yêu", "oai", "oay", "oeo", "uây", "uôi", "ươi", "ươu", "uya", "uyê", "uyu",
}


def _apply_marks(s: str) -> Optional[str]:
    """dd/aa/ee/oo/aw/ow/uw -> đ/â/ê/ô/ă/ơ/ư (chỉ chữ thường a-z)."""
    s = s.replace("dd", "đ")
    out = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two == "aa":
            out.append("â"); i += 2
        elif two == "ee":
            out.append("ê"); i += 2
        elif two == "oo":
            out.append("ô"); i += 2
        elif two == "aw":
            out.append("ă"); i += 2
        elif two == "ow":
            out.append("ơ"); i += 2
        elif two == "uw":
            out.append("ư"); i += 2
        elif s[i] == "w":
            return None  # 'w' lẻ không hợp lệ
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def _split_syllable(s: str):
    """Tách (onset, nucleus, coda) nếu s là âm tiết hợp lệ (chưa có dấu thanh), ngược lại None."""
    vowels = set(_VOWEL_BASE)
    n = len(s)
    best = None
    for on in sorted(ONSETS + ("",), key=len, reverse=True):
        if not s.startswith(on):
            continue
        rest = s[len(on):]
        if on == "gi" and rest == "":
            continue  # "gi" một mình = g + i
        j = 0
        while j < len(rest) and rest[j] in vowels:
            j += 1
        nucleus, coda = rest[:j], rest[j:]
        if not nucleus or nucleus not in NUCLEI:
            continue
        if coda and coda not in CODAS:
            continue
        if on == "g" and nucleus[0] in "ie":
            continue  # gi/ghe: g + i/e không hợp lệ
        if on == "" and n == 0:
            continue
        best = (on, nucleus, coda)
        break
    if best is None and s.startswith("gi") and len(s) == 2:
        return ("g", "i", "")
    return best


def _tone_position(nucleus: str, coda: str) -> int:
    marked = [i for i, ch in enumerate(nucleus) if ch in "ăâêôơư"]
    if marked:
        if "ơ" in nucleus:
            return nucleus.index("ơ")
        return marked[-1]
    if len(nucleus) == 1:
        return 0
    if len(nucleus) == 2:
        if coda or nucleus in ("oa", "oe", "uy"):
            return 1
        return 0
    return 1


def _put_tone(onset: str, nucleus: str, coda: str, tone: int) -> str:
    if tone:
        pos = _tone_position(nucleus, coda)
        base = nucleus[pos]
        nucleus = nucleus[:pos] + _TONED[base][tone] + nucleus[pos + 1:]
    return onset + nucleus + coda


def decode_telex_word(word: str) -> Optional[str]:
    """Giải mã một từ chữ thường a-z. Trả về từ tiếng Việt, hoặc None nếu không phải Telex hợp lệ."""
    if not re.fullmatch(r"[a-z]{3,}", word):
        return None
    candidates = [(word, 0)]
    # phím dấu ở cuối từ, hoặc sau phần vần nhưng trước nguyên âm cuối ("nhieefu": dấu f trước u)
    for k in range(1, len(word)):
        if word[k] not in TONE_KEYS:
            continue
        after_vowel = word[k - 1] in _VOWEL_BASE or word[k - 1] in "aeiouy"
        at_end_after_coda = k == len(word) - 1 and word[:k].endswith(("ch", "ng", "nh", "c", "m", "n", "p", "t"))
        # Phím dấu phải đứng ngay sau nguyên âm (kể cả khi còn nguyên âm cuối: "nhieefu") hoặc ở cuối sau phụ âm cuối.
        # Phím dấu sau phụ âm đầu ("tra", "tro", "triệu": r sau t) là chữ thường của từ, không phải dấu.
        if (after_vowel and re.fullmatch(r"[aeiouy]*", word[k + 1:])) or at_end_after_coda:
            candidates.append((word[:k] + word[k + 1:], _TONE_INDEX[word[k]]))
    has_telex_marker = bool(re.search(r"aa|ee|oo|dd|aw|ow|uw", word)) or len(candidates) > 1
    if not has_telex_marker:
        return None
    for stem, tone in candidates[1:] + candidates[:1]:
        marked = _apply_marks(stem)
        if marked is None:
            continue
        syl = _split_syllable(marked)
        if syl is None:
            continue
        onset, nucleus, coda = syl
        if coda in STOP_CODAS and tone not in (0, 1, 5):
            continue
        out = _put_tone(onset, nucleus, coda, tone)
        if out != word:
            return out
    return None


# Từ ASCII ngắn hay gặp trong câu hỏi kinh doanh mà luật Telex sẽ giải mã nhầm ("max" -> "mã")
COMMON_ASCII_TOKENS = {"max", "min", "six", "gas", "bar", "box", "tax", "mix", "fax", "pass"}


def decode_telex_text(text: str, protected: Optional[Iterable[str]] = None) -> str:
    """
    Giải mã các từ Telex thô trong câu. Bỏ qua: chuỗi trong ngoặc kép, từ viết hoa giữa câu,
    từ nằm trong `protected` (đã bỏ dấu, chữ thường: tên mặt hàng, tên cột... của shop).
    """
    prot: Set[str] = {p.lower() for p in (protected or [])} | COMMON_ASCII_TOKENS
    parts = re.split(r'("[^"]*"|\'[^\']*\')', text)
    for pi in range(0, len(parts), 2):
        seg = parts[pi]

        def _sub(m: "re.Match[str]") -> str:
            w = m.group(0)
            if not w.islower():
                # Chỉ từ đầu câu viết hoa kiểu "Tieenf ..." được xét; từ viết hoa giữa câu là tên riêng/ký hiệu
                if not (w[:1].isupper() and w[1:].islower() and m.start() == 0 and pi == 0):
                    return w
            lw = w.lower()
            if lw in prot:
                return w
            dec = decode_telex_word(lw)
            if dec is None:
                return w
            return dec.capitalize() if w[:1].isupper() else dec

        parts[pi] = re.sub(r"[A-Za-z]+", _sub, seg)
    return "".join(parts)
