# -*- coding: utf-8 -*-
"""
Module 4: Execution Sandbox & Self-Correction Feedback Loop.
Thực thi đoạn mã Python do LLM sinh ra trong một tiến trình con (subprocess) cô lập
với giới hạn thời gian (timeout mặc định 10 giây), cung cấp sẵn:
  - `tables`: dict[str, pd.DataFrame] chứa toàn bộ các bảng đã chuẩn hóa của shop.
  - `con`: kết nối DuckDB in-memory đã đăng ký toàn bộ bảng.
  - `as_of_date`: chuỗi 'YYYY-MM-DD' của câu hỏi.
  - `date_context`: dict thông tin thời gian từ DateResolver.
  - `pd`, `np`, `duckdb`, `re`, `math`, `datetime`, `date`.
Nếu có lỗi cú pháp hoặc Runtime Exception, trích xuất Traceback ngắn gọn để LLM tự sửa lỗi.
"""

import ast
import concurrent.futures
import json
import math
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
    "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "Exception",
    "FloatingPointError", "IndexError", "KeyError", "LookupError", "NameError",
    "OverflowError", "RuntimeError", "StopIteration", "SyntaxError", "TypeError",
    "ValueError", "ZeroDivisionError", "SystemExit",
}


def make_safe_import(duckdb_proxy: Any, pd_mod: Any, np_mod: Any) -> Any:
    """Tạo hàm import an toàn chỉ cho phép các thư viện cần thiết, chặn đứng việc gọi hệ thống ngoài."""
    allowed = {
        "math": math,
        "re": re,
        "json": json,
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
        "unicodedata": unicodedata,
        "duckdb": duckdb_proxy,
        "pandas": pd_mod,
        "pd": pd_mod,
        "numpy": np_mod,
        "np": np_mod,
    }

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".")[0].lower()
        if root not in allowed:
            raise ImportError(f"Importing module '{name}' is forbidden for security in DataMate Sandbox.")
        if root == "datetime":
            import datetime as _dt
            return _dt
        return allowed[root]

    return _safe_import


def make_safe_builtins(custom_import_fn: Any) -> Dict[str, Any]:
    """Tạo whitelist __builtins__ loại bỏ hoàn toàn các hàm nguy hiểm (open, eval, exec, compile...)."""
    import builtins
    b_dict: Dict[str, Any] = {}
    for name in SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            b_dict[name] = getattr(builtins, name)
    b_dict["__import__"] = custom_import_fn
    return b_dict


def has_while_loop(code: str) -> bool:
    """Kiểm tra mã nguồn có chứa vòng lặp while hay không qua AST để tránh rò rỉ thread in-process."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                return True
    except Exception:
        pass
    return False


def strip_accents(text: Any) -> str:
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip()
    s = s.replace("đ", "d").replace("Đ", "d")
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def _serialize_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()[:10]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if type(obj).__name__ == "Decimal":
        # DuckDB trả DECIMAL (vd `100.0 * x`, cột DECIMAL) -> phải là số JSON, không phải chuỗi "3.75"
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def _build_col_fuzzy_rules(tables: Dict[str, pd.DataFrame]) -> List[Tuple[Any, str]]:
    special_cols = sorted(
        {str(c) for df in tables.values() for c in df.columns if re.search(r"[^\w]", str(c))},
        key=len,
        reverse=True,
    )
    accent_map = {
        "a": "[aàáảãạăằắẳẵặâầấẩẫậ]", "e": "[eèéẻẽẹêềếểễệ]", "i": "[iìíỉĩị]",
        "o": "[oòóỏõọôồốổỗộơờớởỡợ]", "u": "[uùúủũụưừứửữự]", "y": "[yỳýỷỹỵ]",
        "d": "[dđ]"
    }
    col_fuzzy_rules = []
    for col in special_cols:
        clean_toks = [t for t in re.split(r"[\s_()/,.-]+", col) if t]
        if not clean_toks:
            continue
        pat_toks = []
        for tok in clean_toks:
            chars = []
            for ch in tok.lower():
                base = "d" if ch in "dđ" else ("a" if ch in "aàáảãạăằắẳẵặâầấẩẫậ" else ("e" if ch in "eèéẻẽẹêềếểễệ" else ("i" if ch in "iìíỉĩị" else ("o" if ch in "oòóỏõọôồốổỗộơờớởỡợ" else ("u" if ch in "uùúủũụưừứửữự" else ("y" if ch in "yỳýỷỹỵ" else ch))))))
                chars.append(accent_map.get(base, re.escape(ch)))
            pat_toks.append("".join(chars))
        sep = r"[\s_()/,.-]*"
        trail = r"\)?" if col.strip().endswith(")") else ""
        lead = r"\(?" if col.strip().startswith("(") else ""
        pat_regex = re.compile(r"(?<![\w\"`])" + lead + sep.join(pat_toks) + trail + r"(?![\w\"`])", re.IGNORECASE)
        col_fuzzy_rules.append((pat_regex, f'"{col}"'))
    return col_fuzzy_rules


class _CopyOnAccessTables(dict):
    """dict bảng trả về bản sao sâu của DataFrame ở lần truy cập đầu (lazy) để code sinh ra không làm hỏng cache."""

    def __init__(self, shared: Dict[str, pd.DataFrame]) -> None:
        super().__init__(shared)
        self._copies: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self._copies:
            self._copies[key] = super().__getitem__(key).copy(deep=True)
        return self._copies[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def values(self):  # type: ignore[override]
        return [self[k] for k in self.keys()]

    def items(self):  # type: ignore[override]
        return [(k, self[k]) for k in self.keys()]


class SafeDuckDBWrapper:
    """Bọc kết nối DuckDB để tự động chuyển MySQL backticks thành double-quotes, so khớp mờ tên cột và phòng thủ cột giả định."""

    def __init__(
        self,
        inner_con: Any,
        col_fuzzy_rules: Optional[List[Tuple[Any, str]]] = None,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self._inner = inner_con
        self._rules = col_fuzzy_rules or []
        self._tables = tables or {}

    def _fix_sql(self, sql: str) -> str:
        fixed = re.sub(r"`([^`]+)`", r'"\1"', str(sql))
        if self._rules:
            parts = re.split(r"('[^']*'|\"[^\"]*\")", fixed)
            for idx in range(0, len(parts), 2):
                chunk = parts[idx]
                for pat_re, repl in self._rules:
                    chunk = pat_re.sub(repl, chunk)
                parts[idx] = chunk
            fixed = "".join(parts)

        # Phòng thủ cột giả định (Binder Error Prevention):
        if self._tables:
            all_cols = {str(c).lower() for df in self._tables.values() if hasattr(df, "columns") for c in df.columns}
            # 1. status_norm
            if "status_norm" not in all_cols:
                fixed = re.sub(r'CASE\s+WHEN\s+("status_norm"|\bstatus_norm\b)\s*=\s*[\'"][^\'"]*[\'"]\s+THEN\s+([^E]+?)\s+END', r'NULL', fixed, flags=re.I)
                fixed = re.sub(r'COALESCE\s*\(\s*(?:\w+\.)?("status_norm"|\bstatus_norm\b)\s*,\s*\'[^\']*\'\s*\)\s*(?:NOT\s+IN\s*\([^)]*\)|IN\s*\([^)]*\)|<>\s*\'[^\']*\'|!=\s*\'[^\']*\'|=\s*\'[^\']*\')', '1=1', fixed, flags=re.I)
                fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s*=\s*[\'"][^\'"]*[\'"]', '1=1', fixed, flags=re.I)
                fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s*!=\s*[\'"][^\'"]*[\'"]', '1=1', fixed, flags=re.I)
                fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s+NOT\s+IN\s*\([^)]*\)', '1=1', fixed, flags=re.I)
                fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s+IN\s*\([^)]*\)', '1=1', fixed, flags=re.I)
            # 2. revenue_norm
            if "revenue_norm" not in all_cols:
                repl_rev = '"Thành tiền"' if "thành tiền" in all_cols else ('"Tổng tiền"' if "tổng tiền" in all_cols else '"revenue_norm"')
                if repl_rev != '"revenue_norm"':
                    fixed = re.sub(r'COALESCE\s*\(\s*("revenue_norm"|\brevenue_norm\b)\s*,\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*\)', repl_rev, fixed, flags=re.I)
                    fixed = re.sub(r'COALESCE\s*\(\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*,\s*("revenue_norm"|\brevenue_norm\b)\s*\)', r'\1', fixed, flags=re.I)
                    fixed = re.sub(r'("revenue_norm"|\brevenue_norm\b)', repl_rev, fixed, flags=re.I)
            # 3. qty_base
            if "qty_base" not in all_cols:
                repl_qty = '"Số lượng"' if "số lượng" in all_cols else ('"SL"' if "sl" in all_cols else '"qty_base"')
                if repl_qty != '"qty_base"':
                    fixed = re.sub(r'COALESCE\s*\(\s*("qty_base"|\bqty_base\b)\s*,\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*\)', repl_qty, fixed, flags=re.I)
                    fixed = re.sub(r'COALESCE\s*\(\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*,\s*("qty_base"|\bqty_base\b)\s*\)', r'\1', fixed, flags=re.I)
                    fixed = re.sub(r'("qty_base"|\bqty_base\b)', repl_qty, fixed, flags=re.I)

        return fixed

    def execute(self, sql: str, *args, **kwargs):
        return self._inner.execute(self._fix_sql(sql), *args, **kwargs)

    def sql(self, query: str, *args, **kwargs):
        return self._inner.sql(self._fix_sql(query), *args, **kwargs)

    def query(self, query: str, *args, **kwargs):
        return self._inner.query(self._fix_sql(query), *args, **kwargs)

    def close(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _FuzzyDateContext(dict):
    """Tra khoá mờ trong date_context. Khoá không xác định được -> KeyError (để vòng tự sửa lỗi kích hoạt),
    KHÔNG âm thầm trả về một kỳ khác (vd 'last_month' từng bị trả thành tháng này)."""

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        k_low = str(key).lower().replace(" ", "_").replace("-", "_")
        for orig_k in self:
            if orig_k.lower().replace(" ", "_") == k_low:
                return super().__getitem__(orig_k)
        is_prev = any(t in k_low for t in ("last", "prev", "previous", "truoc", "trước", "ago", "rồi", "roi"))
        if "two_week" in k_low or "2_week" in k_low or "hai_tuan" in k_low or "2_tuan" in k_low:
            if "two_weeks_ago" in self:
                return super().__getitem__("two_weeks_ago")
        elif "week" in k_low or "tuan" in k_low or "tuần" in k_low:
            target = "last_week" if is_prev else "this_week"
            if target in self:
                return super().__getitem__(target)
        elif "month" in k_low or "thang" in k_low or "tháng" in k_low:
            target = ("prev_month_same_days" if "same" in k_low else "prev_month_full") if is_prev else "this_month"
            if target in self:
                return super().__getitem__(target)
        elif k_low in ("today", "now", "hom_nay"):
            return super().__getitem__("today")
        raise KeyError(f"date_context không có khoá {key!r}. Các khoá hợp lệ: {sorted(k for k in self.keys() if k.isascii())}")


SANDBOX_RUNNER_TEMPLATE = r'''# -*- coding: utf-8 -*-
import json
import math
import re
import sys
import traceback
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent_root_placeholder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import BusinessDataStore

def strip_accents(text):
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip()
    s = s.replace("đ", "d").replace("Đ", "d")
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()

def _serialize_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()[:10]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if type(obj).__name__ == "Decimal":
        # DuckDB trả DECIMAL (vd `100.0 * x`, cột DECIMAL) -> phải là số JSON, không phải chuỗi "3.75"
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if pd.isna(obj):
        return None
    return str(obj)

class _FuzzyDateContext(dict):
    """Tra khoá mờ trong date_context. Khoá không xác định được -> KeyError (để vòng tự sửa lỗi kích hoạt),
    KHÔNG âm thầm trả về một kỳ khác (vd 'last_month' từng bị trả thành tháng này)."""

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        k_low = str(key).lower().replace(" ", "_").replace("-", "_")
        for orig_k in self:
            if orig_k.lower().replace(" ", "_") == k_low:
                return super().__getitem__(orig_k)
        is_prev = any(t in k_low for t in ("last", "prev", "previous", "truoc", "trước", "ago", "rồi", "roi"))
        if "two_week" in k_low or "2_week" in k_low or "hai_tuan" in k_low or "2_tuan" in k_low:
            if "two_weeks_ago" in self:
                return super().__getitem__("two_weeks_ago")
        elif "week" in k_low or "tuan" in k_low or "tuần" in k_low:
            target = "last_week" if is_prev else "this_week"
            if target in self:
                return super().__getitem__(target)
        elif "month" in k_low or "thang" in k_low or "tháng" in k_low:
            target = ("prev_month_same_days" if "same" in k_low else "prev_month_full") if is_prev else "this_month"
            if target in self:
                return super().__getitem__(target)
        elif k_low in ("today", "now", "hom_nay"):
            return super().__getitem__("today")
        raise KeyError(f"date_context không có khoá {key!r}. Các khoá hợp lệ: {sorted(k for k in self.keys() if k.isascii())}")

def main():
    payload_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    business_id = payload["business_id"]
    as_of_date = payload["as_of_date"]
    date_context = _FuzzyDateContext(payload.get("date_context", {}))
    data_root = Path(payload["data_root"])
    cache_root = Path(payload["cache_root"]) if payload.get("cache_root") else None
    code_str = payload["code"]

    store = BusinessDataStore(data_root=data_root, cache_root=cache_root)
    raw_con, tables = store.create_duckdb_connection(business_id)

    special_cols = sorted(
        {str(c) for df in tables.values() for c in df.columns if re.search(r"[^\w]", str(c))},
        key=len,
        reverse=True,
    )

    accent_map = {
        "a": "[aàáảãạăằắẳẵặâầấẩẫậ]", "e": "[eèéẻẽẹêềếểễệ]", "i": "[iìíỉĩị]",
        "o": "[oòóỏõọôồốổỗộơờớởỡợ]", "u": "[uùúủũụưừứửữự]", "y": "[yỳýỷỹỵ]",
        "d": "[dđ]"
    }
    col_fuzzy_rules = []
    for col in special_cols:
        clean_toks = [t for t in re.split(r"[\s_()/,.-]+", col) if t]
        if not clean_toks:
            continue
        pat_toks = []
        for tok in clean_toks:
            chars = []
            for ch in tok.lower():
                base = "d" if ch in "dđ" else ("a" if ch in "aàáảãạăằắẳẵặâầấẩẫậ" else ("e" if ch in "eèéẻẽẹêềếểễệ" else ("i" if ch in "iìíỉĩị" else ("o" if ch in "oòóỏõọôồốổỗộơờớởỡợ" else ("u" if ch in "uùúủũụưừứửữự" else ("y" if ch in "yỳýỷỹỵ" else ch))))))
                chars.append(accent_map.get(base, re.escape(ch)))
            pat_toks.append("".join(chars))
        sep = r"[\s_()/,.-]*"
        trail = r"\)?" if col.strip().endswith(")") else ""
        lead = r"\(?" if col.strip().startswith("(") else ""
        pat_regex = re.compile(r"(?<![\w\"`])" + lead + sep.join(pat_toks) + trail + r"(?![\w\"`])", re.IGNORECASE)
        col_fuzzy_rules.append((pat_regex, f'"{col}"'))

    class _SafeDuckDBWrapper:
        def __init__(self, inner_con, tables=None):
            self._inner = inner_con
            self._tables = tables or {}

        def _fix_sql(self, sql: str) -> str:
            # 1. Tự động chuyển dấu backtick kiểu MySQL `Tên cột` sang dấu ngoặc kép chuẩn DuckDB "Tên cột"
            fixed = re.sub(r"`([^`]+)`", r'"\1"', str(sql))
            # 2. Tự động bọc ngoặc kép quanh các tên cột có dấu cách / biến thể bỏ dấu nằm ngoài chuỗi '...' hoặc "..."
            parts = re.split(r"('[^']*'|\"[^\"]*\")", fixed)
            for idx in range(0, len(parts), 2):
                chunk = parts[idx]
                for pat_re, repl in col_fuzzy_rules:
                    chunk = pat_re.sub(repl, chunk)
                parts[idx] = chunk
            fixed = "".join(parts)

            if self._tables:
                all_cols = {str(c).lower() for df in self._tables.values() if hasattr(df, "columns") for c in df.columns}
                if "status_norm" not in all_cols:
                    fixed = re.sub(r'CASE\s+WHEN\s+("status_norm"|\bstatus_norm\b)\s*=\s*[\'"][^\'"]*[\'"]\s+THEN\s+([^E]+?)\s+END', r'NULL', fixed, flags=re.I)
                    fixed = re.sub(r'COALESCE\s*\(\s*(?:\w+\.)?("status_norm"|\bstatus_norm\b)\s*,\s*\'[^\']*\'\s*\)\s*(?:NOT\s+IN\s*\([^)]*\)|IN\s*\([^)]*\)|<>\s*\'[^\']*\'|!=\s*\'[^\']*\'|=\s*\'[^\']*\')', '1=1', fixed, flags=re.I)
                    fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s*=\s*[\'"][^\'"]*[\'"]', '1=1', fixed, flags=re.I)
                    fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s*!=\s*[\'"][^\'"]*[\'"]', '1=1', fixed, flags=re.I)
                    fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s+NOT\s+IN\s*\([^)]*\)', '1=1', fixed, flags=re.I)
                    fixed = re.sub(r'("status_norm"|\bstatus_norm\b)\s+IN\s*\([^)]*\)', '1=1', fixed, flags=re.I)
                if "revenue_norm" not in all_cols:
                    repl_rev = '"Thành tiền"' if "thành tiền" in all_cols else ('"Tổng tiền"' if "tổng tiền" in all_cols else '"revenue_norm"')
                    if repl_rev != '"revenue_norm"':
                        fixed = re.sub(r'COALESCE\s*\(\s*("revenue_norm"|\brevenue_norm\b)\s*,\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*\)', repl_rev, fixed, flags=re.I)
                        fixed = re.sub(r'COALESCE\s*\(\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*,\s*("revenue_norm"|\brevenue_norm\b)\s*\)', r'\1', fixed, flags=re.I)
                        fixed = re.sub(r'("revenue_norm"|\brevenue_norm\b)', repl_rev, fixed, flags=re.I)
                if "qty_base" not in all_cols:
                    repl_qty = '"Số lượng"' if "số lượng" in all_cols else ('"SL"' if "sl" in all_cols else '"qty_base"')
                    if repl_qty != '"qty_base"':
                        fixed = re.sub(r'COALESCE\s*\(\s*("qty_base"|\bqty_base\b)\s*,\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*\)', repl_qty, fixed, flags=re.I)
                        fixed = re.sub(r'COALESCE\s*\(\s*("[^"]+"|[a-zA-Z0-9_\u00C0-\u024F\u1EA0-\u1EF9]+)\s*,\s*("qty_base"|\bqty_base\b)\s*\)', r'\1', fixed, flags=re.I)
                        fixed = re.sub(r'("qty_base"|\bqty_base\b)', repl_qty, fixed, flags=re.I)
            return fixed

        def execute(self, sql: str, *args, **kwargs):
            return self._inner.execute(self._fix_sql(sql), *args, **kwargs)

        def sql(self, query: str, *args, **kwargs):
            return self._inner.sql(self._fix_sql(query), *args, **kwargs)

        def query(self, query: str, *args, **kwargs):
            return self._inner.query(self._fix_sql(query), *args, **kwargs)

        def close(self, *args, **kwargs):
            pass

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    con = _SafeDuckDBWrapper(raw_con, tables=tables)
    # Monkey-patch duckdb.connect/sql/execute/query để nếu LLM tự gọi duckdb.connect() thì vẫn dùng kết nối an toàn đã có đủ bảng
    duckdb.connect = lambda *a, **kw: con
    duckdb.sql = con.sql
    duckdb.execute = con.execute
    duckdb.query = con.query

    SAFE_BUILTIN_NAMES = {
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
        "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None", "NotImplemented", "Ellipsis",
        "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "Exception",
        "FloatingPointError", "IndexError", "KeyError", "LookupError", "NameError",
        "OverflowError", "RuntimeError", "StopIteration", "SyntaxError", "TypeError",
        "ValueError", "ZeroDivisionError", "SystemExit",
    }

    allowed_import = {
        "math": math, "re": re, "json": json, "datetime": datetime, "date": date,
        "timedelta": timedelta, "unicodedata": unicodedata, "duckdb": duckdb,
        "pandas": pd, "pd": pd, "numpy": np, "np": np,
    }

    def _safe_runner_import(name, *args, **kwargs):
        root = name.split(".")[0].lower()
        if root not in allowed_import:
            raise ImportError(f"Importing module '{name}' is forbidden for security in DataMate Sandbox.")
        if root == "datetime":
            import datetime as _dt
            return _dt
        return allowed_import[root]

    import builtins as _b_mod
    runner_builtins = {}
    for b_name in SAFE_BUILTIN_NAMES:
        if hasattr(_b_mod, b_name):
            runner_builtins[b_name] = getattr(_b_mod, b_name)
    runner_builtins["__import__"] = _safe_runner_import

    env = {
        "__builtins__": runner_builtins,
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "re": re,
        "math": math,
        "date": date,
        "datetime": datetime,
        "timedelta": timedelta,
        "unicodedata": unicodedata,
        "strip_accents": strip_accents,
        "con": con,
        "tables": tables,
        "as_of_date": as_of_date,
        "date_context": date_context,
        "result": None,
        "answer": None,
    }

    try:
        exec(compile(code_str, "<llm_generated_code>", "exec"), env, env)
        raw_res = env.get("result")
        if raw_res is None and env.get("answer") is not None:
            raw_res = env.get("answer")
        if raw_res is None and callable(env.get("solve")):
            raw_res = env["solve"](tables, con, as_of_date)

        # Nếu kết quả là DuckDB Relation, chuyển về DataFrame trước
        if hasattr(raw_res, "df") and callable(getattr(raw_res, "df")):
            try:
                raw_res = raw_res.df()
            except Exception:
                pass

        # Nếu kết quả là DataFrame hoặc Series, chuyển đổi sang cấu trúc chuẩn
        if isinstance(raw_res, pd.DataFrame):
            raw_res = {
                "columns": [str(c) for c in raw_res.columns],
                "rows": raw_res.values.tolist(),
            }
        elif isinstance(raw_res, pd.Series):
            raw_res = raw_res.tolist()

        out_data = {"ok": True, "result": raw_res, "error": None}
    except Exception as exc:
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        user_frames = [
            line for line in tb_lines[:-1]
            if "<llm_generated_code>" in line
        ]
        tb_str = "".join(user_frames + [tb_lines[-1]]).strip()
        code_lines = code_str.splitlines()
        line_nums = [int(x) for x in re.findall(r'File "<llm_generated_code>", line (\d+)', tb_str)]
        offending_info = ""
        if line_nums:
            ln_no = line_nums[-1]
            if 1 <= ln_no <= len(code_lines):
                offending_info = f"\nDòng gây lỗi (line {ln_no}): {code_lines[ln_no - 1].strip()}"
        ordered_tbls = sorted(
            [(t, d) for t, d in tables.items() if not re.match(r"^.+?_(\d+|202[4-9])$", t)],
            key=lambda item: (0 if item[0] in code_str else 1, item[0]),
        )
        schema_hint = "; ".join(
            f"{t}: { [c for c in d.columns if c != '_sheet_name'] }"
            for t, d in ordered_tbls
        )
        out_data = {
            "ok": False,
            "result": None,
            "error": f"{tb_str}{offending_info}\nCác bảng và cột hợp lệ trong `tables` (ưu tiên bảng đang dùng): {schema_hint}",
        }

    output_path.write_text(
        json.dumps(out_data, ensure_ascii=False, default=_serialize_default),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
'''


def extract_python_code(llm_response: str) -> str:
    """Trích xuất khối mã Python từ phản hồi Markdown của LLM (kể cả khi thiếu dấu đóng ``` hoặc có lời dẫn)."""
    if not llm_response:
        return ""
    # 1. Ưu tiên khối ```python ... ``` hoặc ```py ... ``` đã đóng hoàn chỉnh
    py_matches = re.findall(r"```(?:python|py)\s*\n(.*?)```", llm_response, flags=re.DOTALL | re.IGNORECASE)
    if py_matches:
        return py_matches[-1].strip()

    # 1b. Khối ``` ... ``` bất kỳ đã đóng hoàn chỉnh nếu không có thẻ python rõ ràng
    any_matches = re.findall(r"```\s*\n(.*?)```", llm_response, flags=re.DOTALL)
    if any_matches:
        # Lọc các block trông giống code python (chứa import, =, return, def, con, tables, df)
        for cand in reversed(any_matches):
            c_str = cand.strip()
            if any(k in c_str for k in ("=", "import ", "result", "con.", "tables[", "def ")):
                return c_str
        return any_matches[-1].strip()

    # 2. Nếu có mở ```python nhưng bị cắt trước dấu đóng ```
    open_match = re.search(r"```(?:python|py)?\s*\n(.*)$", llm_response, flags=re.DOTALL | re.IGNORECASE)
    if open_match:
        return open_match.group(1).replace("```", "").strip()

    # 3. Loại bỏ các dòng lời dẫn tiếng Việt ở đầu trước dòng code Python đầu tiên
    lines = llm_response.strip().splitlines()
    code_start = 0
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith(("#", "import ", "from ", "df", "hd", "ct", "tables", "con", "result", "def ")):
            code_start = idx
            break
    return "\n".join(lines[code_start:]).replace("```", "").strip()


from src.ingest import BusinessDataStore


class _DuckDBModuleProxy:
    """Proxy cho module duckdb trong sandbox in-process, định tuyến an toàn tới kết nối đã nạp đủ bảng."""

    def __init__(self, safe_con: Any) -> None:
        self._con = safe_con

    def connect(self, *args, **kwargs):
        return self._con

    def sql(self, *args, **kwargs):
        return self._con.sql(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self._con.execute(*args, **kwargs)

    def query(self, *args, **kwargs):
        return self._con.query(*args, **kwargs)

    def close(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        return getattr(duckdb, name)


class PythonSandbox:
    """Thực thi mã Python an toàn và siêu tốc (in-process mặc định <0.05s, tự động fallback subprocess nếu cần)."""

    def __init__(
        self,
        data_root: Path,
        cache_root: Optional[Path] = None,
        timeout_seconds: float = 10.0,
        python_executable: Optional[str] = None,
        in_process: bool = True,
        store: Optional[BusinessDataStore] = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_root = Path(data_root)
        self.cache_root = cache_root or (self.project_root / ".cache" / "ingested")
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.in_process = in_process
        self._warm_store = store or BusinessDataStore(data_root=self.data_root, cache_root=self.cache_root)

    def _execute_in_process(
        self,
        cleaned_code: str,
        business_id: str,
        as_of_date: str,
        date_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_con, shared_tables = self._warm_store.create_duckdb_connection(business_id)
        # Code LLM chỉ thấy bản sao: DataFrame cache dùng chung cho mọi câu hỏi/luồng không được bị sửa tại chỗ
        tables = _CopyOnAccessTables(shared_tables)
        rules = _build_col_fuzzy_rules(shared_tables)
        safe_con = SafeDuckDBWrapper(raw_con, rules, tables=shared_tables)
        duckdb_proxy = _DuckDBModuleProxy(safe_con)

        fuzzy_ctx = _FuzzyDateContext(date_context or {})

        safe_import = make_safe_import(duckdb_proxy, pd, np)
        builtins_dict = make_safe_builtins(safe_import)

        env = {
            "__builtins__": builtins_dict,
            "pd": pd,
            "np": np,
            "duckdb": duckdb_proxy,
            "re": re,
            "math": math,
            "date": date,
            "datetime": datetime,
            "timedelta": timedelta,
            "unicodedata": unicodedata,
            "strip_accents": strip_accents,
            "con": safe_con,
            "tables": tables,
            "as_of_date": as_of_date,
            "date_context": fuzzy_ctx,
            "result": None,
            "answer": None,
        }

        worker_tid = [None]

        def _target():
            worker_tid[0] = threading.get_ident()
            try:
                compiled = compile(cleaned_code, "<llm_generated_code>", "exec")
                exec(compiled, env, env)
                raw_res = env.get("result")
                if raw_res is None and env.get("answer") is not None:
                    raw_res = env.get("answer")
                if raw_res is None and callable(env.get("solve")):
                    raw_res = env["solve"](tables, safe_con, as_of_date)

                if hasattr(raw_res, "df") and callable(getattr(raw_res, "df")):
                    try:
                        raw_res = raw_res.df()
                    except Exception:
                        pass

                if isinstance(raw_res, pd.DataFrame):
                    raw_res = {
                        "columns": [str(c) for c in raw_res.columns],
                        "rows": raw_res.values.tolist(),
                    }
                elif isinstance(raw_res, pd.Series):
                    raw_res = raw_res.tolist()

                # Chuẩn hóa JSON để tương thích 100% với các kiểu dữ liệu của subprocess
                raw_res = json.loads(json.dumps(raw_res, default=_serialize_default, ensure_ascii=False))
                return {"ok": True, "result": raw_res, "error": None}
            except SystemExit:
                return {"ok": False, "result": None, "error": "Terminated on timeout."}

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = executor.submit(_target)
        try:
            res = fut.result(timeout=self.timeout_seconds)
            executor.shutdown(wait=False)
            return res
        except concurrent.futures.TimeoutError:
            # 1. Hủy truy vấn DuckDB ở mức C++ engine nếu đang thực thi dở
            try:
                raw_con.interrupt()
            except Exception:
                pass

            # 2. Đóng executor không block luồng chính (không dùng ctypes để tránh hỏng bộ nhớ/crash CPython)
            executor.shutdown(wait=False, cancel_futures=True)
            return {
                "ok": False,
                "result": None,
                "error": f"TimeoutError: Code execution exceeded {self.timeout_seconds}s limit.",
            }
        except Exception as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            user_frames = [
                line for line in tb_lines[:-1]
                if "<llm_generated_code>" in line
            ]
            tb_str = "".join(user_frames + [tb_lines[-1]]).strip()
            code_lines = cleaned_code.splitlines()
            line_nums = [int(x) for x in re.findall(r'File "<llm_generated_code>", line (\d+)', tb_str)]
            offending_info = ""
            if line_nums:
                ln_no = line_nums[-1]
                if 1 <= ln_no <= len(code_lines):
                    offending_info = f"\nDòng gây lỗi (line {ln_no}): {code_lines[ln_no - 1].strip()}"
            ordered_tbls = sorted(
                [(t, d) for t, d in shared_tables.items() if not re.match(r"^.+?_(\d+|202[4-9])$", t)],
                key=lambda item: (0 if item[0] in cleaned_code else 1, item[0]),
            )
            schema_hint = "; ".join(
                f"{t}: { [c for c in d.columns if c != '_sheet_name'] }"
                for t, d in ordered_tbls
            )
            return {
                "ok": False,
                "result": None,
                "error": f"{tb_str}{offending_info}\nCác bảng và cột hợp lệ trong `tables` (ưu tiên bảng đang dùng): {schema_hint}",
            }

    def _execute_subprocess(
        self,
        cleaned_code: str,
        business_id: str,
        as_of_date: str,
        date_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._warm_store.load_tables(business_id, use_cache=True)
        with tempfile.TemporaryDirectory(prefix="datamate_sb_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            runner_path = tmp_path / "sandbox_runner.py"
            payload_path = tmp_path / "payload.json"
            output_path = tmp_path / "output.json"

            runner_script = SANDBOX_RUNNER_TEMPLATE.replace(
                "Path(__file__).resolve().parent_root_placeholder",
                f"Path({repr(str(self.project_root))})",
            )
            runner_path.write_text(runner_script, encoding="utf-8")

            payload = {
                "business_id": business_id,
                "as_of_date": as_of_date,
                "date_context": date_context or {},
                "data_root": str(self.data_root),
                "cache_root": str(self.cache_root) if self.cache_root else None,
                "code": cleaned_code,
            }
            payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            try:
                proc = subprocess.run(
                    [self.python_executable, str(runner_path), str(payload_path), str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "ok": False,
                    "result": None,
                    "error": f"TimeoutError: Code execution exceeded {self.timeout_seconds}s limit.",
                }

            if output_path.exists():
                try:
                    return json.loads(output_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    return {"ok": False, "result": None, "error": f"Output parse error: {exc}"}

            stderr_msg = (proc.stderr or proc.stdout or "Unknown subprocess failure").strip()
            return {"ok": False, "result": None, "error": stderr_msg[-1000:]}

    def execute(
        self,
        code: str,
        business_id: str,
        as_of_date: str,
        date_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy đoạn code Python an toàn (ưu tiên in-process <0.05s, tự động fallback subprocess nếu cần).
        Trả về dict: {"ok": bool, "result": Any, "error": Optional[str]}
        """
        cleaned_code = extract_python_code(code)
        if not cleaned_code:
            return {"ok": False, "result": None, "error": "Empty Python code generated."}

        # Kiểm tra cú pháp nhanh
        try:
            compile(cleaned_code, "<llm_generated_code>", "exec")
        except SyntaxError as syn_err:
            return {
                "ok": False,
                "result": None,
                "error": f"SyntaxError at line {syn_err.lineno}: {syn_err.msg}\n{syn_err.text or ''}",
            }

        if self.in_process:
            # Nếu phát hiện vòng lặp while, chuyển sang subprocess để bảo đảm an toàn triệt để
            # (tiến trình con có thể bị OS terminate/SIGKILL khi timeout, không làm rò rỉ thread in-process)
            if has_while_loop(cleaned_code):
                return self._execute_subprocess(
                    cleaned_code=cleaned_code,
                    business_id=business_id,
                    as_of_date=as_of_date,
                    date_context=date_context,
                )
            try:
                return self._execute_in_process(
                    cleaned_code=cleaned_code,
                    business_id=business_id,
                    as_of_date=as_of_date,
                    date_context=date_context,
                )
            except Exception:
                # Fallback an toàn sang subprocess nếu có lỗi môi trường cấp thấp
                pass

        return self._execute_subprocess(
            cleaned_code=cleaned_code,
            business_id=business_id,
            as_of_date=as_of_date,
            date_context=date_context,
        )


Sandbox = PythonSandbox

