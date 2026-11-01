# -*- coding: utf-8 -*-
"""
Module 3: Dual-Prompt Consensus Engine (Qwen2.5-Coder-14B-Instruct-AWQ) & Multi-Stage Agent.
Kết hợp:
  - Module 1 (`BusinessDataStore` trong `src/ingest.py`)
  - Module 2 (`DateResolver` trong `src/date_resolver.py` & `RuleEngine` trong `src/rule_engine.py`)
  - Module 3 (`QwenCoderClient` với mô hình Qwen2.5-Coder-14B-AWQ & Dual-Perspective Prompts)
  - Module 4 (`PythonSandbox` trong `src/sandbox.py` với vòng lặp tự sửa lỗi)
  - Module 5 (`postprocess_and_guard` & `check_consensus` trong `src/postprocess.py`)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.date_resolver import DateResolver
from src.ingest import BusinessDataStore
from src.postprocess import check_consensus, postprocess_and_guard
from src.rule_engine import DAILFewShotSelector, DynamicEntityGrounder, RuleEngine
from src.sandbox import PythonSandbox


DEFAULT_QWEN_14B_AWQ_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
DEFAULT_QWEN_7B_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OLLAMA_7B_MODEL = "qwen2.5-coder:7b"
DEFAULT_MODEL = DEFAULT_QWEN_14B_AWQ_MODEL

# -----------------------------------------------------------------------------
# PROMPT 1: Tư duy Đại số quan hệ (SQL-First / Relational Set-Based via DuckDB)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_SQL = """Bạn là Kỹ sư Dữ liệu chuyên về SQL và DuckDB giải quyết bài toán DataMate (Zalo AI Challenge 2026).
Nhiệm vụ: Viết một đoạn mã Python truy vấn dữ liệu từ kết nối DuckDB `con` (đã đăng ký sẵn toàn bộ bảng dữ liệu) để trả lời câu hỏi.

QUY TẮC CỐT LÕI (DUCKDB SQL):
1. Ưu tiên giải bài toán bằng truy vấn `con.execute(\"\"\"SELECT ...\"\"\")`:
   - Dữ liệu đã nạp sẵn trong kết nối `con` (DuckDB) và `tables` (Pandas), TUYỆT ĐỐI KHÔNG gọi `duckdb.connect(...)` hoặc mở file mới.
   - Tuyệt đối KHÔNG viết comment (# ...) trong code Python, chỉ viết trực tiếp các lệnh thực thi.
   - Luôn bọc `COALESCE(SUM(...), 0)` ở các hàm tổng hợp để tránh trả về NULL. Khi lấy kết quả: `row = con.execute(query).fetchone(); val = row[0] if row else 0; result = {"type": "number", "answer": float(val) if val is not None else 0.0}`.
   - Tất cả tên bảng và tên cột tiếng Việt có dấu, khoảng trắng BẮT BUỘC bọc trong ngoặc kép: `"Thành tiền"`, `"Mã đơn hàng"`, `"Ngày"`, `"Trạng thái"`.
   - Lọc ngày: `WHERE CAST("Ngày" AS DATE) BETWEEN DATE 'start' AND DATE 'end'` (hoặc `CAST("Ngày" AS VARCHAR)`).
   - Lọc đơn hợp lệ (bỏ đơn hủy, đơn trả): `WHERE "status_norm" = 'Hoàn thành'` (hoặc `"Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%trả%'`).
   - Doanh thu (revenue): Luôn là `SUM(COALESCE("revenue_norm", "Thành tiền"))` (tuyệt đối không nhân số lượng với ANY_VALUE của đơn giá!).
   - Xếp hạng ngày hoặc khách có doanh thu cao nhất: `ORDER BY SUM(COALESCE("revenue_norm", "Thành tiền")) DESC LIMIT N`.
   - Xếp hạng SẢN PHẨM bán chạy nhất: `ORDER BY SUM(COALESCE("qty_base", "Số lượng")) DESC LIMIT N` (xếp theo sản lượng bán ra).
   - Đếm đơn hàng: `SELECT COUNT(DISTINCT "Mã đơn hàng")` trên bảng chi tiết; không dùng `COUNT(*)` kẻo đếm nhầm dòng sản phẩm.
   - Xếp hạng khách hàng: BẮT BUỘC loại bỏ khách lẻ và khách vãng lai: `WHERE "Khách hàng" NOT ILIKE '%khách lẻ%' AND "Khách hàng" NOT ILIKE '%lẻ%' AND "Khách hàng" NOT ILIKE '%vãng lai%' AND "Khách hàng" IS NOT NULL`.
   - Khi JOIN nhiều bảng (vd DonHang và ChiTietDon), luôn chỉ định rõ alias trước tên cột (vd: o."Khách hàng", c."Khách hàng") để tránh lỗi DuckDB "Ambiguous reference to column name".
   - Tỉ lệ đơn hủy: KHÔNG được lọc bỏ đơn hủy trước khi tính (tuyệt đối không dùng `WHERE "status_norm" = 'Hoàn thành'`). Công thức: `COUNT(DISTINCT CASE WHEN "status_norm" = 'Đã hủy' THEN "Mã đơn hàng" END) * 100.0 / NULLIF(COUNT(DISTINCT "Mã đơn hàng"), 0)`.
   - Tỉ trọng doanh thu nhóm/sản phẩm: Mẫu số phải là tổng doanh thu toàn shop (không dùng INNER JOIN làm co hẹp mẫu số). Ví dụ: `SUM(CASE WHEN "Nhóm" = '...' THEN "Thành tiền" ELSE 0 END) * 100.0 / NULLIF(SUM("Thành tiền"), 0)`.
   - Tìm kiếm chuỗi tiếng Việt không dấu hoặc viết tắt: Dùng `WHERE strip_accents("Tên sản phẩm") ILIKE strip_accents('%từ khóa%')` (DuckDB hỗ trợ hàm `strip_accents()`).
   - Hỏi số lượng ("bao nhiêu đĩa / cái / kg / ly / chai / lon / nồi"): Trả về SỐ LƯỢNG (cột `"Số lượng"` hoặc `"qty_base"`), KHÔNG phải doanh thu!
   - Cột "business_id" đã có sẵn trong mọi bảng, dữ liệu trong DuckDB đã thuộc riêng shop hiện tại.
2. Định dạng biến kết quả cuối cùng:
   - `number`: `result = {"type": "number", "answer": float(val)}`
   - `percent`: `result = {"type": "percent", "answer": float(val)}` (đã nhân 100, ví dụ: 15.5)
   - `date`: `result = {"type": "date", "answer": "YYYY-MM-DD"}`
   - `string`: `result = {"type": "string", "answer": "Tên đối tượng"}`
   - `list`: `result = {"type": "list", "answer": [r[0] for r in con.execute(...).fetchall()]}` (danh sách phẳng các chuỗi/số)
   - `table`: `result = {"columns": ["tên cột 1", "tên cột 2"], "rows": [list(r) for r in con.execute(...).fetchall()]}`
   - `unanswerable`: `result = {"type": "unanswerable", "reason": "..."}`
3. Chỉ trả về một khối mã ```python ... ``` duy nhất, không giải thích văn bản thừa."""

# -----------------------------------------------------------------------------
# PROMPT 2: Tư duy Phòng thủ từng bước với DuckDB CTE (Defensive CTE Engine)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_DEFENSIVE_SQL = """Bạn là Chuyên gia Khoa học Dữ liệu chú trọng tính chính xác và kiểm soát rủi ro biên (DataMate 2026).
Nhiệm vụ: Viết một đoạn mã Python sử dụng kết nối DuckDB `con` với kỹ thuật Common Table Expressions (WITH ... AS ...) để bóc tách bài toán thành từng bước phòng thủ chặt chẽ.

QUY TẮC CỐT LÕI (DEFENSIVE CTE DUCKDB RELATIONAL):
1. Tổ chức truy vấn kiểm soát dữ liệu chặt chẽ:
   - CTE 1 (valid_orders): Lọc đơn hàng hợp lệ theo ngày và loại bỏ đơn 'Đã hủy', 'Trả hàng' (CHÚ Ý: nếu tính tỉ lệ hủy đơn thì giữ lại đơn hủy để đếm). BẮT BUỘC dùng `SELECT * FROM ... WHERE ...` để giữ lại toàn bộ các cột phục vụ cho các bước JOIN và GROUP BY phía sau, tuyệt đối không liệt kê thiếu cột gây lỗi Binder Error.
   - CTE 2 (filtered_data): Nếu câu hỏi cần thêm thông tin từ bảng thứ hai (chi tiết hàng hoặc danh mục), ghép nối theo khóa ngoại (`Mã đơn hàng`, `Mã sản phẩm`). TUYỆT ĐỐI KHÔNG tự ý JOIN bảng với chính nó hoặc JOIN thừa thãi khi câu hỏi chỉ cần dữ liệu từ 1 bảng duy nhất (như bảng bán hàng hoặc bảng hóa đơn), vì JOIN thừa sẽ nhân bản dòng dữ liệu (Cartesian product) làm sai lệch nghiêm trọng kết quả tính toán!
   - CTE 3 (aggregated): Nhóm và tính toán tổng tiền, số lượng (`qty_base`), hoặc tỉ trọng.
   - Hoàn toàn có thể viết trực tiếp câu lệnh SELECT duy nhất nếu truy vấn đơn giản hoặc đơn bảng, không bắt buộc phải tách nhiều CTE.
2. Quy tắc an toàn bắt buộc:
   - Dữ liệu đã nạp sẵn trong kết nối `con` (DuckDB) và `tables` (Pandas), TUYỆT ĐỐI KHÔNG gọi `duckdb.connect(...)` hoặc mở file mới.
   - Tuyệt đối KHÔNG viết comment (# ...) trong code Python, chỉ viết trực tiếp các lệnh thực thi.
   - Luôn bọc `COALESCE(SUM(...), 0)` ở các hàm tổng hợp để tránh trả về NULL; trong Python: `row = con.execute(query).fetchone(); val = row[0] if row else 0; result = {"type": "number", "answer": float(val) if val is not None else 0.0}`.
   - Tên bảng và tên cột tiếng Việt có dấu, khoảng trắng BẮT BUỘC bọc trong ngoặc kép: `"Thành tiền"`, `"Mã đơn hàng"`, `"Ngày"`, `"Trạng thái"`.
   - Lọc ngày theo khoảng thời gian: `WHERE CAST("Ngày" AS DATE) BETWEEN DATE 'start' AND DATE 'end'`.
   - Lọc lũy kế ("tính đến ngày X", "tính đến hết ngày X", "còn nợ", "còn tồn"): Lấy toàn bộ giao dịch phát sinh từ đầu dữ liệu đến hết ngày X: `WHERE CAST("Ngày" AS DATE) <= DATE 'X'` (hoặc `WHERE "Ngày" <= 'YYYY-MM-DD 23:59:59'`), TUYỆT ĐỐI KHÔNG lọc gói gọn trong riêng ngày X (`BETWEEN X AND X`).
   - Công nợ khách hàng (còn nợ, nợ nhiều nhất): Tính `SUM(COALESCE("Tổng tiền hàng", "Tổng bill", "Thành tiền", 0)) - SUM(COALESCE("Khách đã trả", "Khách trả", 0))` hoặc cột nợ phát sinh, KHÔNG lấy tổng doanh thu làm công nợ.
   - Doanh thu (revenue): Luôn là `SUM(COALESCE("revenue_norm", "Thành tiền"))`.
   - Sản phẩm bán chạy nhất: Luôn xếp theo `SUM(COALESCE("qty_base", "Số lượng")) DESC` (sản lượng bán ra), KHÔNG xếp theo doanh thu.
   - Tránh chia cho 0: Luôn dùng `NULLIF(..., 0)` ở mẫu số để tránh lỗi crash division by zero.
   - So khớp chuỗi tiếng Việt không dấu: Dùng `strip_accents("Tên sản phẩm") ILIKE strip_accents('%từ khóa%')`.
   - Bỏ khách lẻ và khách vãng lai khi xếp hạng khách hàng: `WHERE "Khách hàng" NOT ILIKE '%khách lẻ%' AND "Khách hàng" NOT ILIKE '%lẻ%' AND "Khách hàng" NOT ILIKE '%vãng lai%' AND "Khách hàng" IS NOT NULL`.
3. Định dạng biến kết quả cuối cùng:
   - `number`: `result = {"type": "number", "answer": float(val)}`
   - `percent`: `result = {"type": "percent", "answer": float(val)}` (đã nhân 100, ví dụ: 40.3)
   - `date`: `result = {"type": "date", "answer": "YYYY-MM-DD"}`
   - `string`: `result = {"type": "string", "answer": "Tên đối tượng"}`
   - `list`: `result = {"type": "list", "answer": [r[0] for r in con.execute(...).fetchall()]}` (danh sách phẳng các chuỗi/số)
   - `table`: `result = {"columns": ["cột 1", "cột 2"], "rows": [list(r) for r in con.execute(...).fetchall()]}`
   - `unanswerable`: `result = {"type": "unanswerable", "reason": "..."}`
4. Chỉ trả về một khối mã ```python ... ``` duy nhất, không giải thích văn bản thừa."""

SYSTEM_PROMPT_DEFENSIVE = SYSTEM_PROMPT_DEFENSIVE_SQL

# Mặc định tương thích ngược
SYSTEM_PROMPT = SYSTEM_PROMPT_SQL


class QwenCoderClient:
    """
    Client kết nối tới mô hình Qwen2.5-Coder (hỗ trợ cả 14B-AWQ và 7B).
    Hỗ trợ 3 chế độ backend:
      - 'openai': Gọi qua OpenAI-compatible API (vLLM continuous batching).
      - 'ollama': Gọi trực tiếp qua Ollama API.
      - 'transformers': Nạp trực tiếp bằng HuggingFace Transformers.
      - 'custom': Hàm callback mock phục vụ kiểm thử.
    """

    def __init__(
        self,
        backend: str = "openai",
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        request_timeout: float = 90.0,
        custom_generator: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.backend = backend.lower().strip()
        if model_name:
            self.model_name = model_name
        elif self.backend == "ollama":
            self.model_name = DEFAULT_OLLAMA_7B_MODEL
        else:
            self.model_name = DEFAULT_MODEL

        self.api_base = (
            api_base
            or os.environ.get("QWEN_API_BASE")
            or ("http://localhost:11434" if self.backend == "ollama" else "http://localhost:8000/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.custom_generator = custom_generator

        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                raise_on_status=False,
            ),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._hf_tokenizer: Any = None
        self._hf_model: Any = None

    def _get_openai_url(self) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def health_check(self, timeout: float = 5.0) -> bool:
        """Kiểm tra kết nối tới LLM backend trước khi chạy toàn bộ pipeline."""
        if self.custom_generator is not None:
            return True
        if self.backend == "openai":
            try:
                models_url = self._get_openai_url().replace("/chat/completions", "/models")
                headers = {"Authorization": f"Bearer {self.api_key}"}
                r = self._session.get(models_url, headers=headers, timeout=timeout)
                return r.status_code == 200
            except Exception:
                return False
        if self.backend == "ollama":
            try:
                r = self._session.get(f"{self.api_base}/api/tags", timeout=timeout)
                return r.status_code == 200
            except Exception:
                return False
        return True

    def _ensure_hf_loaded(self) -> None:
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._hf_model.eval()

    def generate(self, messages: List[Dict[str, str]], seed: int = 42) -> str:
        """Sinh phản hồi từ danh sách messages theo chuẩn ChatML với tính tất định cao."""
        if self.custom_generator is not None:
            return self.custom_generator(messages)

        if self.backend == "openai":
            url = self._get_openai_url()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            # Chỉ truyền seed khi sampling (temperature > 0) để tương thích vLLM OpenAI API
            if seed is not None and self.temperature > 0:
                payload["seed"] = seed

            last_exc: Optional[Exception] = None
            for call_attempt in range(3):
                try:
                    resp = self._session.post(url, json=payload, headers=headers, timeout=self.request_timeout)
                    if resp.status_code == 404:
                        # Tự động đồng bộ model name từ /v1/models nếu tên model không khớp
                        try:
                            models_url = url.replace("/chat/completions", "/models")
                            m_resp = self._session.get(models_url, headers=headers, timeout=5.0)
                            if m_resp.ok:
                                m_data = m_resp.json()
                                available = [m["id"] for m in m_data.get("data", []) if "id" in m]
                                if available and self.model_name not in available:
                                    self.model_name = available[0]
                                    payload["model"] = self.model_name
                                    resp = self._session.post(url, json=payload, headers=headers, timeout=self.request_timeout)
                        except Exception:
                            pass
                    if resp.status_code >= 400:
                        try:
                            err_body = resp.json()
                            err_msg = err_body.get("message") or err_body.get("error", {}).get("message") or resp.text
                        except Exception:
                            err_msg = resp.text
                        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {err_msg}")
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                except Exception as exc:
                    last_exc = exc
                    if call_attempt < 2:
                        time.sleep(1.0 * (call_attempt + 1))
            raise RuntimeError(f"OpenAI API call to {url} failed after 3 attempts: {last_exc}")

        if self.backend == "ollama":
            url = f"{self.api_base}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            if seed is not None and self.temperature > 0:
                payload["options"]["seed"] = seed
            resp = self._session.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

        if self.backend == "transformers":
            self._ensure_hf_loaded()
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)
            import torch  # type: ignore
            torch.manual_seed(seed)
            with torch.no_grad():
                generated_ids = self._hf_model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    temperature=None,
                )
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            return self._hf_tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

        raise ValueError(f"Unsupported backend: {self.backend}")


class DataMateAgent:
    """
    Pipeline Multi-Stage Dual-Prompt Consensus Agent:
      1. Ingestion & Schema Clean (Calamine + Generic Key Mapping)
      2. Dynamic Rulebook Knob Reader & Dynamic Industry Selector
      3. Dual-Perspective Code Generator (Prompt 1: SQL-Centric, Prompt 2: Defensive Procedural)
      4. Python Sandbox & Isolated Self-Correction Feedback Loop
      5. Strict Consensus Gate & Decimal Precision Guard
    """

    def __init__(
        self,
        data_root: Path,
        calendar_path: Path,
        rulebook_path: Path,
        llm_client: QwenCoderClient,
        cache_root: Optional[Path] = None,
        sandbox_timeout: float = 10.0,
        max_retries: int = 2,
        verbose: bool = False,
        consensus_mode: str = "dual",
        arbitration_strategy: str = "strict",
    ) -> None:
        self.store = BusinessDataStore(data_root=data_root, cache_root=cache_root)
        self.date_resolver = DateResolver(calendar_path=calendar_path)
        self.rule_engine = RuleEngine(rulebook_path=rulebook_path)
        self.entity_grounder = DynamicEntityGrounder()
        self.few_shot_selector = DAILFewShotSelector()
        self.sandbox = PythonSandbox(
            data_root=data_root,
            cache_root=cache_root,
            timeout_seconds=sandbox_timeout,
            store=self.store,
        )
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.verbose = verbose
        self.consensus_mode = consensus_mode.lower().strip()
        self.arbitration_strategy = arbitration_strategy.lower().strip()

    def build_user_prompt(
        self,
        question_obj: Dict[str, Any],
        expected_type: str,
        date_prompt: str,
        rules_prompt: str,
        schema_prompt: str,
        entity_hints: str = "",
        few_shot_prompt: str = "",
    ) -> str:
        question = question_obj["question"]
        as_of_date = question_obj["as_of_date"]
        business_id = question_obj["business_id"]

        parts = [
            schema_prompt,
            f"=== LUẬT NGHIỆP VỤ LIÊN QUAN ===\n{rules_prompt}",
            f"=== CÂU HỎI CẦN GIẢI QUYẾT ===\n"
            f"- Mã câu hỏi: {question_obj.get('id', 'unknown')}\n"
            f"- Cửa hàng (`business_id`): {business_id}\n"
            f"- Ngày hiện tại (`as_of_date` - Hôm nay): {as_of_date}\n"
            f"- Câu hỏi: {question}\n"
            f"- Kiểu đáp án kỳ vọng (`expected_type`): {expected_type}",
            f"=== NGỮ CẢNH THỜI GIAN ĐÃ QUY ĐỔI ===\n{date_prompt}",
        ]
        if entity_hints:
            parts.append(entity_hints)
        if few_shot_prompt:
            parts.append(few_shot_prompt)
        parts.extend([
            "=== YÊU CẦU SINH MÃ PYTHON ===\n"
            "Hãy viết mã Python truy vấn dữ liệu từ `tables` hoặc `con` DuckDB để tính ra kết quả chính xác.\n"
            "Toàn bộ bảng dữ liệu đã nạp sẵn vào biến `con` và `tables`, không gọi `duckdb.connect()`, không viết comment (# ...).\n"
            f"Cuối đoạn code, bắt buộc gán `result = {{\"type\": \"...\", \"answer\": ...}}`."
        ])
        return "\n\n".join(parts)

    def _solve_single_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        qid: str,
        business_id: str,
        as_of_date: str,
        date_ctx: Dict[str, Any],
        expected_type: str,
        question: str,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], str, Any]:
        """Thực thi một luồng prompt độc lập kèm vòng lặp tự sửa lỗi nếu crash."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_code = ""
        last_raw = None
        last_error = ""
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                llm_reply = self.llm_client.generate(messages, seed=seed)
                last_code = llm_reply
            except Exception as llm_exc:
                last_error = f"LLM Generation Error: {llm_exc}"
                print(f"    ⚠️ [LLM Error] {qid} ({business_id}, attempt {attempt+1}/{total_attempts}): {llm_exc}", flush=True)
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return {"id": qid, "type": "abstain", "error": last_error}, last_code, None

            exec_outcome = self.sandbox.execute(
                code=llm_reply,
                business_id=business_id,
                as_of_date=as_of_date,
                date_context=date_ctx,
            )
            last_raw = exec_outcome.get("result")

            if exec_outcome.get("ok") and last_raw is not None:
                final_pred = postprocess_and_guard(
                    question_id=qid,
                    question=question,
                    raw_output=last_raw,
                    expected_type=expected_type,
                )
                return final_pred, last_code, last_raw

            err_msg = exec_outcome.get("error") or "Biến `result` có giá trị `None` sau khi chạy code."
            last_error = f"Sandbox Error: {err_msg}"
            if attempt < self.max_retries:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt[:8000]},
                    {"role": "assistant", "content": llm_reply[:1500]},
                    {
                        "role": "user",
                        "content": (
                            f"Đoạn mã Python trên gặp lỗi khi chạy trong Sandbox:\n"
                            f"```text\n{err_msg[:1200]}\n```\n"
                            f"Hãy kiểm tra lại đúng tên bảng và tên cột chính xác trong schema, "
                            f"sau đó viết lại khối ```python ... ``` sửa lỗi hoàn chỉnh."
                        ),
                    },
                ]

        return {"id": qid, "type": "abstain", "error": last_error}, last_code, last_raw

    def answer_question_dual(
        self, question_obj: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Giải quyết câu hỏi bằng kiến trúc Dual-Prompt Consensus Gate & Tiered Arbitration:
          - Prompt 1: SQL-Centric trên DuckDB (Seed 42).
          - Prompt 2: Defensive Step-by-Step trên Pandas (Seed 100).
          - Phân cấp trọng tài tin cậy:
              * Cả hai đều ra đáp án: Bắt buộc đồng thuận (check_consensus) để tránh sai (-1).
              * Một bên thành công vượt qua guard, một bên lỗi kỹ thuật: Chấp nhận đáp án hợp lệ (Adaptive).
              * Cả hai lỗi: Quyết định Abstain an toàn (0 điểm).
        """
        qid = question_obj["id"]
        business_id = question_obj["business_id"]
        question = question_obj["question"]
        as_of_date = question_obj["as_of_date"]

        try:
            biz_readme = self.store.get_readme(business_id)
            tables = self.store.load_tables(business_id)
            entity_hints = self.entity_grounder.find_grounding_hints(question, business_id, tables)
            few_shot_prompt = self.few_shot_selector.select_exemplars(
                question=question,
                has_entity_hint=bool(entity_hints and entity_hints.strip()),
            )
            expected_type = self.rule_engine.infer_expected_type(question)
            date_ctx = self.date_resolver.resolve(question, as_of_date)
            date_prompt = self.date_resolver.format_for_prompt(question, as_of_date)
            rules_prompt = self.rule_engine.select_relevant_rules(question, biz_readme)
            schema_prompt = self.store.build_compact_schema_prompt(business_id, question=question)

            is_unans, unans_reason = self.rule_engine.detect_unanswerable_deterministic(
                question_obj=question_obj,
                store=self.store,
                date_resolver=self.date_resolver,
            )
            if is_unans:
                unans_resp = {"id": qid, "type": "unanswerable", "reason": unans_reason}
                details = {
                    "id": qid,
                    "business_id": business_id,
                    "question": question,
                    "expected_type": expected_type,
                    "prompt1_code": "# Deterministic unanswerable detected",
                    "prompt1_raw": unans_resp,
                    "prompt1_pred": unans_resp,
                    "prompt2_code": "# Deterministic unanswerable detected",
                    "prompt2_raw": unans_resp,
                    "prompt2_pred": unans_resp,
                    "is_consensus": True,
                    "decision_reason": "deterministic_unanswerable",
                    "final_decision": "unanswerable",
                    "final_answer": None,
                }
                return unans_resp, details
        except Exception as e:
            abstain_resp = {"id": qid, "type": "abstain"}
            details = {
                "id": qid,
                "business_id": business_id,
                "question": question,
                "expected_type": "unknown",
                "error": str(e),
                "is_consensus": False,
                "decision_reason": "exception_before_prompt",
                "final_decision": "abstain",
                "final_answer": None,
            }
            return abstain_resp, details

        user_prompt = self.build_user_prompt(
            question_obj=question_obj,
            expected_type=expected_type,
            date_prompt=date_prompt,
            rules_prompt=rules_prompt,
            schema_prompt=schema_prompt,
            entity_hints=entity_hints,
            few_shot_prompt=few_shot_prompt,
        )

        def _is_meaningful_pred(pred: Any) -> bool:
            if not isinstance(pred, dict):
                return False
            ptype = pred.get("type")
            if ptype not in ("number", "percent", "date", "string", "list", "table", "boolean", "unanswerable"):
                return False
            if ptype == "abstain":
                return False
            if ptype == "unanswerable":
                return True
            ans = pred.get("answer")
            if ans is None:
                return False
            if isinstance(ans, str) and (not ans.strip() or ans.strip().lower() in ("none", "nan", "null")):
                return False
            if isinstance(ans, (list, tuple)) and len(ans) == 0:
                return False
            return True

        # Nếu cấu hình single prompt mode (chỉ Prompt 1)
        if self.consensus_mode == "single":
            pred1, code1, raw1 = self._solve_single_prompt(
                system_prompt=SYSTEM_PROMPT_SQL,
                user_prompt=user_prompt,
                qid=qid,
                business_id=business_id,
                as_of_date=as_of_date,
                date_ctx=date_ctx,
                expected_type=expected_type,
                question=question,
                seed=42,
            )
            p1_valid = _is_meaningful_pred(pred1)
            final_pred = pred1 if p1_valid else {"id": qid, "type": "abstain"}
            details = {
                "id": qid,
                "business_id": business_id,
                "question": question,
                "expected_type": expected_type,
                "prompt1_code": code1,
                "prompt1_raw": raw1,
                "prompt1_pred": pred1,
                "prompt2_code": "",
                "prompt2_raw": None,
                "prompt2_pred": None,
                "is_consensus": p1_valid,
                "decision_reason": "single_prompt_mode",
                "final_decision": final_pred.get("type"),
                "final_answer": final_pred.get("answer"),
            }
            return final_pred, details

        # Chạy Prompt 1 (SQL-Centric) và Prompt 2 (Defensive CTE SQL) ĐỒNG THỜI qua ThreadPoolExecutor
        # Nhờ continuous batching của vLLM, cả 2 prompt được sinh song song trên GPU, giảm 50% thời gian chờ LLM
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(
                self._solve_single_prompt,
                system_prompt=SYSTEM_PROMPT_SQL,
                user_prompt=user_prompt,
                qid=qid,
                business_id=business_id,
                as_of_date=as_of_date,
                date_ctx=date_ctx,
                expected_type=expected_type,
                question=question,
                seed=42,
            )
            fut2 = executor.submit(
                self._solve_single_prompt,
                system_prompt=SYSTEM_PROMPT_DEFENSIVE_SQL,
                user_prompt=user_prompt,
                qid=qid,
                business_id=business_id,
                as_of_date=as_of_date,
                date_ctx=date_ctx,
                expected_type=expected_type,
                question=question,
                seed=100,
            )
            pred1, code1, raw1 = fut1.result()
            pred2, code2, raw2 = fut2.result()

        p1_valid = _is_meaningful_pred(pred1)
        p2_valid = _is_meaningful_pred(pred2)

        # 3. Phân cấp đồng thuận & Trọng tài tin cậy (Tiered Consensus Arbitration)
        is_consensus = False
        decision_reason = ""
        final_pred = {"id": qid, "type": "abstain"}

        if p1_valid and p2_valid:
            # Cả hai bên đều ra đáp án -> Bắt buộc kiểm tra đồng thuận
            if check_consensus(pred1, pred2):
                is_consensus = True
                if pred1.get("type") == "list":
                    a1 = pred1.get("answer") or []
                    a2 = pred2.get("answer") or []
                    final_pred = pred1 if len(a1) >= len(a2) else pred2
                else:
                    final_pred = pred1
                decision_reason = "consensus_match"
            else:
                # Query-Feature Arbitration (Paper Lee & Lee 2024 / Borchmann 2025):
                # Khi câu hỏi hỏi số lượng đơn hàng, nếu P1 dùng COUNT(DISTINCT ...) để chống nhân đôi dòng
                # từ bảng chi tiết trong khi P2 dùng COUNT(*) đơn thuần -> Ưu tiên P1 với độ tin cậy tuyệt đối.
                p1_code_str = str(code1 or "")
                p2_code_str = str(code2 or "")
                q_low = question.lower()
                is_order_cnt_q = any(k in q_low for k in ["mấy đơn", "may don", "nhiêu đơn", "nhieu don", "bao nhiêu đơn", "bao nhiu don", "bn đơn", "bn don", "số đơn", "so don", "mấy hóa đơn", "mấy phiếu", "mấy gd"])
                p1_has_distinct = "COUNT(DISTINCT" in p1_code_str.upper()
                p2_has_distinct = "COUNT(DISTINCT" in p2_code_str.upper()

                if pred1.get("type") == "number" and is_order_cnt_q and p1_has_distinct and not p2_has_distinct:
                    is_consensus = False
                    final_pred = pred1
                    decision_reason = "p1_distinct_semantic_arbitration"
                else:
                    is_consensus = False
                    final_pred = {"id": qid, "type": "abstain"}
                    decision_reason = "consensus_disagreement_abstain"
        elif p1_valid and not p2_valid:
            if self.arbitration_strategy == "adaptive":
                # Prompt 1 (SQL) vượt qua toàn bộ guard an toàn, Prompt 2 lỗi kỹ thuật -> Giữ Prompt 1
                is_consensus = False
                final_pred = pred1
                decision_reason = "p1_valid_p2_failed_fallback"
            else:
                final_pred = {"id": qid, "type": "abstain"}
                decision_reason = "strict_requires_both"
        elif p2_valid and not p1_valid:
            # Prompt 1 (SQL) thất bại là tín hiệu cảnh báo rủi ro cao:
            # Dữ liệu thực nghiệm 300 câu chứng minh p2_valid_p1_failed chỉ đạt 29.5% độ chính xác
            # (13 đúng, 31 sai, âm 18 điểm). Do đó, từ chối fallback này và abstain an toàn để bảo vệ điểm.
            final_pred = {"id": qid, "type": "abstain"}
            decision_reason = "p2_only_rejected_abstain"
        else:
            final_pred = {"id": qid, "type": "abstain"}
            decision_reason = "both_failed"

        details = {
            "id": qid,
            "business_id": business_id,
            "question": question,
            "expected_type": expected_type,
            "prompt1_code": code1,
            "prompt1_raw": raw1,
            "prompt1_pred": pred1,
            "prompt2_code": code2,
            "prompt2_raw": raw2,
            "prompt2_pred": pred2,
            "is_consensus": is_consensus,
            "decision_reason": decision_reason,
            "final_decision": final_pred.get("type"),
            "final_answer": final_pred.get("answer"),
        }
        return final_pred, details

    def answer_question(self, question_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Tương thích ngược với interface chuẩn."""
        final_pred, _ = self.answer_question_dual(question_obj)
        return final_pred
