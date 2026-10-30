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

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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
    if pd.isna(obj):
        return None
    return str(obj)

class _FuzzyDateContext(dict):
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        k_low = str(key).lower().replace(" ", "_")
        for orig_k in self:
            if orig_k.lower().replace(" ", "_") == k_low:
                return super().__getitem__(orig_k)
        if "two_week" in k_low or "2_week" in k_low or "hai_tuan" in k_low or "2_tuan" in k_low:
            return self.get("last_week", (self.get("as_of_date"), self.get("as_of_date")))
        if "week" in k_low or "tuan" in k_low:
            return self.get("this_week", (self.get("as_of_date"), self.get("as_of_date")))
        if "month" in k_low or "thang" in k_low:
            return self.get("this_month", (self.get("as_of_date"), self.get("as_of_date")))
        return (self.get("as_of_date"), self.get("as_of_date"))

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
        def __init__(self, inner_con):
            self._inner = inner_con

        @staticmethod
        def _fix_sql(sql: str) -> str:
            # 1. Tự động chuyển dấu backtick kiểu MySQL `Tên cột` sang dấu ngoặc kép chuẩn DuckDB "Tên cột"
            fixed = re.sub(r"`([^`]+)`", r'"\1"', str(sql))
            # 2. Tự động bọc ngoặc kép quanh các tên cột có dấu cách / biến thể bỏ dấu nằm ngoài chuỗi '...' hoặc "..."
            parts = re.split(r"('[^']*'|\"[^\"]*\")", fixed)
            for idx in range(0, len(parts), 2):
                chunk = parts[idx]
                for pat_re, repl in col_fuzzy_rules:
                    chunk = pat_re.sub(repl, chunk)
                parts[idx] = chunk
            return "".join(parts)

        def execute(self, sql: str, *args, **kwargs):
            return self._inner.execute(self._fix_sql(sql), *args, **kwargs)

        def sql(self, query: str, *args, **kwargs):
            return self._inner.sql(self._fix_sql(query), *args, **kwargs)

        def query(self, query: str, *args, **kwargs):
            return self._inner.query(self._fix_sql(query), *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    con = _SafeDuckDBWrapper(raw_con)
    # Monkey-patch duckdb.connect/sql/execute/query để nếu LLM tự gọi duckdb.connect() thì vẫn dùng kết nối an toàn đã có đủ bảng
    duckdb.connect = lambda *a, **kw: con
    duckdb.sql = con.sql
    duckdb.execute = con.execute
    duckdb.query = con.query

    env = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "re": re,
        "math": math,
        "date": date,
        "datetime": datetime,
        "timedelta": timedelta,
        "unicodedata": unicodedata,
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
    # 1. Ưu tiên khối ```python ... ``` đã đóng hoàn chỉnh
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", llm_response, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

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


class PythonSandbox:
    """Thực thi mã Python cô lập trong subprocess với timeout nghiêm ngặt."""

    def __init__(
        self,
        data_root: Path,
        cache_root: Optional[Path] = None,
        timeout_seconds: float = 10.0,
        python_executable: Optional[str] = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_root = Path(data_root)
        self.cache_root = cache_root or (self.project_root / ".cache" / "ingested")
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self._warm_store = BusinessDataStore(data_root=self.data_root, cache_root=self.cache_root)

    def execute(
        self,
        code: str,
        business_id: str,
        as_of_date: str,
        date_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy đoạn code Python trong subprocess cô lập.
        Trả về dict: {"ok": bool, "result": Any, "error": Optional[str]}
        """
        cleaned_code = extract_python_code(code)
        if not cleaned_code:
            return {"ok": False, "result": None, "error": "Empty Python code generated."}

        # Kiểm tra cú pháp nhanh trước khi tạo subprocess
        try:
            compile(cleaned_code, "<llm_generated_code>", "exec")
        except SyntaxError as syn_err:
            return {
                "ok": False,
                "result": None,
                "error": f"SyntaxError at line {syn_err.lineno}: {syn_err.msg}\n{syn_err.text or ''}",
            }

        # Đảm bảo dữ liệu của business_id đã được chuẩn hóa vào Parquet cache trước khi bấm giờ timeout
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
