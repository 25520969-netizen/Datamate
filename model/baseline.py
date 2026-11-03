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
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.date_resolver import DateResolver
from src.ingest import BusinessDataStore
from src.lexicon import load_lexicon
from src.postprocess import check_consensus, postprocess_and_guard
from src.rule_engine import DAILFewShotSelector, DynamicEntityGrounder, RuleEngine
from src.sandbox import PythonSandbox
from src.telex import decode_telex_text
from src.vietnamese_normalizer import VietnameseNormalizer


DEFAULT_QWEN_14B_AWQ_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
DEFAULT_QWEN_7B_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OLLAMA_7B_MODEL = "qwen2.5-coder:7b"
DEFAULT_MODEL = DEFAULT_QWEN_14B_AWQ_MODEL

# -----------------------------------------------------------------------------
# BỐ CỤC PROMPT TỐI ƯU PREFIX CACHING (vLLM --enable-prefix-caching):
#   system  = NGUYÊN TẮC CHUNG + TOÀN BỘ RULEBOOK      -> giống hệt cho MỌI câu hỏi của tập (chỉ prefill 1 lần)
#   user    = schema shop -> câu hỏi -> vai trò P1/P2/P3 -> chỉ phần sau cùng khác nhau giữa các lượt sinh
# Không đặt giá trị thay đổi theo câu (as_of_date, câu hỏi) vào system prompt.
# Nguyên tắc chung KHÔNG chứa luật nghiệp vụ của một rulebook cụ thể; luật lấy từ rulebook đính kèm.
# -----------------------------------------------------------------------------
GENERAL_GUIDELINES = """Bạn là kỹ sư dữ liệu viết mã Python + DuckDB SQL trả lời câu hỏi tiếng Việt trên dữ liệu kinh doanh (DataMate, Zalo AI Challenge 2026).

=== NGUYÊN TẮC CHUNG ===
1. LUẬT NGHIỆP VỤ (đơn nào được tính, doanh thu, khách, xếp hạng, công nợ, kho, cách hiểu thời gian, khi nào không trả lời được...) lấy DUY NHẤT từ RULEBOOK bên dưới. Câu hỏi nói khác thì theo câu hỏi. Không tự đặt ra luật ngoài rulebook.
2. "Hôm nay" là `as_of_date` ghi trong yêu cầu; mọi cách nói tương đối tính theo ngày này.
3. Cột chuẩn hoá có sẵn (nếu bảng có): `status_norm` = trạng thái đơn đã chuẩn hoá ('Hoàn thành', 'Đã hủy', 'Trả hàng', 'Đang giao' hoặc giữ giá trị gốc nếu không nhận dạng được); `qty_base` = số lượng quy về đơn vị cơ bản theo bảng quy đổi/README; `Đơn vị cơ bản`; `revenue_norm` = cột tiền của dòng. Đối chiếu nghĩa các cột này với rulebook trước khi dùng.
4. ĐỊNH DẠNG ĐÁP ÁN (đề thi): number/percent là số JSON không đơn vị (percent đã nhân 100); date 'YYYY-MM-DD', câu hỏi theo tháng 'YYYY-MM'; boolean True/False; string là tên nguyên văn trong dữ liệu (đủ dấu, giữ hậu tố như "(M)"); list có thứ tự theo câu hỏi; table {"columns": [...], "rows": [[...]]} với cột đầu là tên đối tượng.
5. XẾP HẠNG (đề thi): giảm dần theo tiêu chí câu hỏi (hoặc mặc định trong rulebook), bằng nhau thì theo MÃ tăng dần: `ORDER BY <tiêu chí> DESC, <mã> ASC`. "Top k" -> LIMIT k; ít hơn k đối tượng thì trả hết.
6. KỸ THUẬT: dữ liệu đã nạp sẵn trong `con` (DuckDB) và `tables` (Pandas), KHÔNG gọi `duckdb.connect(...)`, không mở file. Tên bảng/cột có dấu hoặc khoảng trắng bọc ngoặc kép. Cột ngày là chuỗi ISO 'YYYY-MM-DD[ HH:MM:SS]': lọc `CAST("Ngày" AS DATE) BETWEEN DATE 'a' AND DATE 'b'`, lấy giờ `hour(CAST(... AS TIMESTAMP))`. `COALESCE(SUM(...), 0)` ở hàm tổng hợp, `NULLIF(..., 0)` ở mẫu số. Đếm đơn bằng `COUNT(DISTINCT <mã đơn>)`. So khớp tên không dấu: `strip_accents(...) ILIKE strip_accents('%...%')`.
7. Cuối code gán `result = {"type": "<number|percent|date|boolean|string|list|table|unanswerable>", "answer": ...}`; `unanswerable` kèm `"reason"`.
8. KHÔNG viết bất kỳ chú thích nào: không `#` trong Python, không `--` hay `/* */` trong SQL, không docstring, không giải thích bằng lời. Chỉ trả về MỘT khối ```python ... ``` rồi dừng."""


def build_system_prompt(rulebook_text: str) -> str:
    """System prompt chung cho mọi câu hỏi/mọi vai trò: nguyên tắc + toàn bộ rulebook của tập dữ liệu."""
    if not rulebook_text.strip():
        return GENERAL_GUIDELINES
    return GENERAL_GUIDELINES + "\n\n=== RULEBOOK ===\n" + rulebook_text.strip()


# -----------------------------------------------------------------------------
# VAI TRÒ (đặt CUỐI user prompt để phần trước đó dùng chung prefix cache giữa P1/P2/P3)
# PROMPT 1: Tư duy Đại số quan hệ (SQL-First)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_SQL = """=== CÁCH GIẢI (truy vấn trực tiếp) ===
- Viết một truy vấn DuckDB SQL gọn, trực tiếp; JOIN chỉ khi cần, ghi rõ alias trước tên cột, không JOIN thừa gây nhân bản dòng.
- Tỉ lệ / tỉ trọng: mẫu số đúng như rulebook định nghĩa.
- Hỏi số lượng ("bao nhiêu ly / cái / kg") thì trả số lượng, không phải tiền; câu hỏi nêu size thì lọc đúng size.
- KHÔNG viết bất kỳ chú thích nào."""

# -----------------------------------------------------------------------------
# PROMPT 2: Tư duy Phòng thủ từng bước với DuckDB CTE
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_DEFENSIVE_SQL = """=== CÁCH GIẢI (CTE phòng thủ) ===
- Tách bước bằng CTE (WITH ... AS ...): (a) lọc kỳ thời gian, (b) lọc bản ghi được tính theo rulebook, (c) tổng hợp.
- Không JOIN bảng chi tiết với bảng đơn khi chỉ cần một bảng; nếu cần thông tin cấp đơn thì JOIN với bảng đơn đã DISTINCT theo mã đơn.
- Số còn lại / lũy kế ("tính đến hết ngày X") hay số phát sinh trong kỳ: xác định đúng theo mục thời gian của rulebook.
- KHÔNG viết bất kỳ chú thích nào."""

# -----------------------------------------------------------------------------
# PROMPT 3 (tie-break): phân rã bằng các CTE đặt tên theo bước (ý tưởng decomposition của DIN-SQL),
# KHÔNG viết kế hoạch bằng lời. Chỉ chạy khi P1 và P2 không đồng thuận, làm lá phiếu thứ ba
# (gom nhóm theo kết quả thực thi kiểu Agentar-Scale-SQL: đáp án được >= 2/3 ứng viên ủng hộ mới được nhận).
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_PLAN_SQL = """=== CÁCH GIẢI (phân rã CTE đặt tên) ===
- Viết MỘT truy vấn gồm các CTE đặt tên theo từng bước, theo thứ tự: `ky_hoi` (khoảng thời gian), `ban_ghi_hop_le` (bản ghi được tính theo rulebook), `tong_hop` (phép tính), rồi SELECT cuối. Tên CTE thay cho mọi lời giải thích.
- KHÔNG viết bất kỳ chú thích nào."""

# Ngừng sinh ngay khi khối code đóng: bỏ phần giải thích thừa sau code (tiết kiệm token sinh)
# Chỉ mẫu ĐÓNG khối ("\n```" + xuống dòng); khối mở là "```python" nên không bị cắt nhầm ở đầu
CODE_STOP_SEQUENCES = ["\n```\n"]

SYSTEM_PROMPT_DEFENSIVE = SYSTEM_PROMPT_DEFENSIVE_SQL

# Mặc định tương thích ngược
SYSTEM_PROMPT = SYSTEM_PROMPT_SQL

# Nhãn "khách lẻ" chỉ bị loại khi rulebook đang dùng ghi khách lẻ không phải khách (RuleEngine.walk_in_not_customer)
WALK_IN_CUSTOMER_LABELS = {
    "khách lẻ", "khach le", "lẻ", "le", "kl", "khách vãng lai", "khach vang lai", "vãng lai", "vang lai",
}
EMPTY_ANSWER_LABELS = {"none", "null", "nan", ""}


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
        max_tokens: int = 1024,
        request_timeout: float = 300.0,
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
        env_timeout = os.environ.get("QWEN_REQUEST_TIMEOUT")
        if env_timeout:
            try:
                self.request_timeout = float(env_timeout)
            except ValueError:
                self.request_timeout = request_timeout
        else:
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

    def generate(self, messages: List[Dict[str, str]], seed: int = 42, stop: Optional[List[str]] = None) -> str:
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
            # Cố định seed cho vLLM / OpenAI API để đảm bảo tính tất định cao nhất
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop

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
                        sleep_s = 3.0 * (call_attempt + 1) if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)) else 1.0 * (call_attempt + 1)
                        time.sleep(sleep_s)
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
            if seed is not None:
                payload["options"]["seed"] = seed
            if stop:
                payload["options"]["stop"] = stop
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
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
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
    Pipeline Multi-Stage Consensus Agent:
      1. Ingestion & Schema Clean (Calamine + Generic Key Mapping)
      2. Dynamic Rulebook Knob Reader & Dynamic Industry Selector
      3. Dual-Perspective Code Generator (Prompt 1: SQL-Centric, Prompt 2: Defensive CTE)
      4. Python Sandbox & Isolated Self-Correction Feedback Loop
      5. Strict Consensus Gate; khi bất đồng: lá phiếu thứ ba (Prompt 3, plan-then-code) và chỉ nhận
         đáp án được >= 2 ứng viên ủng hộ (gom nhóm theo kết quả thực thi), còn lại Safe Abstain.
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
        seed: int = 42,
        tiebreak: bool = True,
        rewrite_question: bool = False,
    ) -> None:
        self.store = BusinessDataStore(data_root=data_root, cache_root=cache_root)
        self.date_resolver = DateResolver(calendar_path=calendar_path)
        self.rule_engine = RuleEngine(rulebook_path=rulebook_path)
        self.date_resolver.apply_knobs(self.rule_engine.knobs)
        self.entity_grounder = DynamicEntityGrounder()
        self.few_shot_selector = DAILFewShotSelector()
        self.vietnamese_normalizer = VietnameseNormalizer(llm_client=llm_client)
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
        self.seed = seed
        self.tiebreak = tiebreak
        # Viết lại câu hỏi bằng LLM (DART-SQL) tốn thêm một lượt LLM TUẦN TỰ trước khi sinh code -> mặc định tắt
        self.rewrite_question = rewrite_question
        self.system_prompt_common = build_system_prompt(self.rule_engine.full_rulebook)
        self._entity_values_cache: Dict[str, set] = {}
        self._protected_cache: Dict[str, set] = {}

    def build_user_prompt(
        self,
        question_obj: Dict[str, Any],
        expected_type: str,
        date_prompt: str,
        rules_prompt: str,
        schema_prompt: str,
        entity_hints: str = "",
        few_shot_prompt: str = "",
        original_question: str = "",
    ) -> str:
        question = question_obj["question"]
        as_of_date = question_obj["as_of_date"]
        business_id = question_obj["business_id"]

        question_lines = f"- Câu hỏi: {question}\n"
        if original_question and original_question.strip() != question.strip():
            # DART-SQL: câu viết lại chỉ để khử mơ hồ; câu gốc vẫn là căn cứ khi hai câu lệch nghĩa
            question_lines = (
                f"- Câu hỏi gốc: {original_question}\n"
                f"- Câu hỏi đã chuẩn hoá (tham khảo, nếu lệch nghĩa thì theo câu gốc): {question}\n"
            )
        parts = [
            schema_prompt,
            f"=== CÂU HỎI CẦN GIẢI QUYẾT ===\n"
            f"- Mã câu hỏi: {question_obj.get('id', 'unknown')}\n"
            f"- Cửa hàng (`business_id`): {business_id}\n"
            f"- Ngày hiện tại (`as_of_date` - Hôm nay): {as_of_date}\n"
            + question_lines +
            f"- Kiểu đáp án kỳ vọng (`expected_type`): {expected_type}",
        ]
        if rules_prompt:
            # Chỉ là con trỏ tới các mục liên quan; nội dung luật đầy đủ nằm ở RULEBOOK trong system prompt
            parts.append(f"=== MỤC RULEBOOK LIÊN QUAN NHẤT ===\n{rules_prompt}")
        parts.append(f"=== NGỮ CẢNH THỜI GIAN ĐÃ QUY ĐỔI ===\n{date_prompt}")
        if entity_hints:
            parts.append(entity_hints)
        if few_shot_prompt:
            parts.append(few_shot_prompt)
        if expected_type == "table":
            parts.append(
                "Kiểu table: `result = {\"type\": \"table\", \"answer\": {\"columns\": [\"ten_doi_tuong\", \"so_do\"], \"rows\": [list(r) for r in rows]}}`"
            )
        return "\n\n".join(parts)

    def arbitrate_safe_tournament(
        self,
        question: str,
        as_of_date: str,
        code1: str,
        ans1: Any,
        code2: str,
        ans2: Any,
        rules_text: str = "",
    ) -> Optional[int]:
        """
        Trọng tài LLM so sánh cặp (lấy ý tưởng pairwise selector của Agentar-Scale-SQL, nhưng KHÔNG có
        selector huấn luyện RL như bài báo) kết hợp Safe Abstain.
        Chỉ trả về 1 (chọn P1) hoặc 2 (chọn P2) khi phát hiện rõ ràng một bên vi phạm luật trong rulebook được truyền vào.
        Nếu cả hai bên đều không có vi phạm rõ ràng hoặc không chắc chắn -> Trả về None (kích hoạt Safe Abstain 0 điểm).
        """
        referee_prompt = (
            f"Bạn là trọng tài thẩm định truy vấn dữ liệu. Hãy kiểm tra 2 phương án truy vấn sau:\n"
            f"Câu hỏi: {question}\n"
            f"Ngày hiện tại (as_of_date): {as_of_date}\n\n"
            f"--- Phương án 1 ---\n"
            f"Code:\n{code1[:1500]}\n"
            f"Kết quả: {ans1}\n\n"
            f"--- Phương án 2 ---\n"
            f"Code:\n{code2[:1500]}\n"
            f"Kết quả: {ans2}\n\n"
            f"Luật nghiệp vụ (trích rulebook):\n{rules_text[:4000]}\n\n"
            f"Hãy xem xét: Có phương án nào vi phạm rõ ràng luật trên không?\n"
            f"- Nếu Phương án 1 đúng và Phương án 2 vi phạm: trả về '\\boxed{{1}}'\n"
            f"- Nếu Phương án 2 đúng và Phương án 1 vi phạm: trả về '\\boxed{{2}}'\n"
            f"- Nếu không có vi phạm rõ ràng hoặc không chắc chắn bên nào đúng: BẮT BUỘC trả về '\\boxed{{ABSTAIN}}'."
        )
        try:
            decision = self.llm_client.generate([
                {"role": "system", "content": "Bạn là chuyên gia thẩm định SQL khách quan. Trả lời cực kỳ thận trọng."},
                {"role": "user", "content": referee_prompt}
            ], seed=42)
            if "\\boxed{1}" in decision and "\\boxed{2}" not in decision and "ABSTAIN" not in decision.upper():
                return 1
            if "\\boxed{2}" in decision and "\\boxed{1}" not in decision and "ABSTAIN" not in decision.upper():
                return 2
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Đối chiếu đáp án chuỗi với giá trị ô thật trong dữ liệu (cell grounding hậu kiểm)
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_cell(text: Any) -> str:
        """Chuẩn hoá giống luật chấm chuỗi: NFC, lowercase, ký tự không phải chữ/số -> khoảng trắng."""
        nfc = unicodedata.normalize("NFC", str(text)).lower()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in nfc)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _protected_tokens(self, business_id: str, tables: Dict[str, Any]) -> set:
        """Từ (bỏ dấu, chữ thường) xuất hiện trong tên cột/giá trị chữ của shop: không được giải mã Telex."""
        cached = self._protected_cache.get(business_id)
        if cached is not None:
            return cached
        toks: set = set()
        try:
            for df in tables.values():
                for col in getattr(df, "columns", []):
                    toks.update(re.findall(r"[a-z]{2,}", self.rule_engine.strip_accents(str(col))))
                    if df[col].dtype != object:
                        continue
                    uniq = df[col].dropna().unique()
                    if len(uniq) > 50000:
                        continue
                    for v in uniq:
                        if isinstance(v, str):
                            toks.update(re.findall(r"[a-z]{2,}", self.rule_engine.strip_accents(v)))
        except Exception:
            pass
        self._protected_cache[business_id] = toks
        return toks

    def _entity_values(self, business_id: str) -> set:
        if business_id in self._entity_values_cache:
            return self._entity_values_cache[business_id]
        vals: set = set()
        try:
            tables = self.store.load_tables(business_id)
            for df in tables.values():
                for col in getattr(df, "columns", []):
                    if df[col].dtype != object:
                        continue
                    uniq = df[col].dropna().unique()
                    if len(uniq) > 50000:
                        continue
                    for v in uniq:
                        if isinstance(v, str) and 0 < len(v) <= 150:
                            vals.add(self._norm_cell(v))
        except Exception:
            vals = set()
        self._entity_values_cache[business_id] = vals
        return vals

    def _ground_strings(self, pred: Dict[str, Any], raw: Any, business_id: str) -> Dict[str, Any]:
        """
        Postprocess cắt chú thích trong ngoặc ("Trà (307 ly)" -> "Trà"). Nếu bản đã cắt KHÔNG có trong dữ liệu
        nhưng bản gốc trong raw có ("Bia Tiger (24 lon)") thì khôi phục bản gốc.
        """
        ptype = pred.get("type")
        if ptype not in ("string", "list", "table"):
            return pred
        vals = self._entity_values(business_id)
        if not vals:
            return pred

        raw_strings: List[str] = []

        def _collect(obj: Any, depth: int = 0) -> None:
            if depth > 4 or len(raw_strings) > 500:
                return
            if isinstance(obj, str):
                raw_strings.append(unicodedata.normalize("NFC", obj).strip())
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v, depth + 1)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _collect(v, depth + 1)

        _collect(raw)

        def _fix(s: Any) -> Any:
            if not isinstance(s, str) or self._norm_cell(s) in vals:
                return s
            for r in raw_strings:
                if r != s and r.startswith(s) and self._norm_cell(r) in vals:
                    return r
            return s

        out = dict(pred)
        ans = pred.get("answer")
        if ptype == "string":
            out["answer"] = _fix(ans)
        elif ptype == "list" and isinstance(ans, list):
            out["answer"] = [_fix(x) for x in ans]
        elif ptype == "table" and isinstance(ans, dict) and isinstance(ans.get("rows"), list):
            out["answer"] = dict(ans, rows=[[_fix(r[0])] + list(r[1:]) if r else r for r in ans["rows"]])
        return out

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
        # system_prompt ở đây là khối VAI TRÒ (P1/P2/P3): đặt cuối user prompt; system dùng chung cho mọi câu hỏi
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt_common},
            {"role": "user", "content": f"{user_prompt}\n\n{system_prompt}"},
        ]
        gen_kwargs: Dict[str, Any] = {"stop": CODE_STOP_SEQUENCES} if isinstance(self.llm_client, QwenCoderClient) else {}
        last_code = ""
        last_raw = None
        last_error = ""
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                llm_reply = self.llm_client.generate(messages, seed=seed, **gen_kwargs)
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

            candidate_pred = None
            if exec_outcome.get("ok") and last_raw is not None:
                candidate_pred = postprocess_and_guard(
                    question_id=qid,
                    question=question,
                    raw_output=last_raw,
                    expected_type=expected_type,
                    code=str(llm_reply or ""),
                    reject_walk_in=self.rule_engine.walk_in_not_customer,
                )
                if candidate_pred.get("type") != "abstain":
                    return self._ground_strings(candidate_pred, last_raw, business_id), last_code, last_raw

            if not exec_outcome.get("ok"):
                err_msg = exec_outcome.get("error") or "Lỗi cú pháp hoặc lỗi thực thi trong DuckDB/Sandbox."
            elif candidate_pred is not None and candidate_pred.get("type") == "abstain":
                raw_preview = json.dumps(last_raw, ensure_ascii=False, default=str)[:300]
                err_msg = (
                    f"Mã chạy không lỗi nhưng kết quả không dùng được: `result = {raw_preview}`. "
                    "Nguyên nhân thường gặp: rỗng/None (WHERE lọc mất toàn bộ dữ liệu: sai khoảng ngày, sai so khớp chuỗi ILIKE), "
                    "số âm cho đại lượng không thể âm, trả về 'Khách lẻ' (không phải khách), hoặc sai cấu trúc kiểu đáp án. Hãy sửa lại."
                )
            else:
                err_msg = "Biến `result` có giá trị `None` sau khi chạy code."

            last_error = f"Sandbox Error: {err_msg}"
            if attempt < self.max_retries:
                messages.append({"role": "assistant", "content": llm_reply})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Đoạn mã Python/SQL trên gặp lỗi khi chạy trong cơ sở dữ liệu:\n"
                        f"```text\n{err_msg[:1200]}\n```\n"
                        f"Hãy phân tích nguyên nhân lỗi (sai tên bảng/cột, sai cú pháp ngày, rỗng dữ liệu) "
                        f"và viết lại khối lệnh ```python ... ``` hoàn chỉnh để khắc phục chính xác lỗi trên."
                    ),
                })

        return {"id": qid, "type": "abstain", "error": last_error}, last_code, last_raw

    @staticmethod
    def _extract_top_k(question: str) -> Optional[int]:
        """Số phần tử k của câu hỏi xếp hạng: 'top 5', '3 món bán chạy nhất', '5 khách mua nhiều tiền nhất'."""
        q = unicodedata.normalize("NFC", question or "").lower()
        m = re.search(r"\btop\s*(\d{1,2})\b", q)
        if m:
            return int(m.group(1))
        m = re.search(
            r"(?<![\d/.,])\b(\d{1,2})\s+(?:mặt hàng|món|sản phẩm|sp|khách hàng|khách|nhân viên|nhà cung cấp|ncc|thuốc|"
            r"nhóm hàng|nhóm|chi nhánh|thợ|hàng|loại)\b[^?]*?(?:nhất|nhiều|cao|lớn|chạy|đầu)",
            q,
        )
        if m:
            return int(m.group(1))
        return None

    def _is_meaningful_pred(self, pred: Any) -> bool:
        if not isinstance(pred, dict):
            return False
        ptype = pred.get("type")
        if ptype not in ("number", "percent", "date", "string", "list", "table", "boolean", "unanswerable"):
            return False
        if ptype == "unanswerable":
            return True
        ans = pred.get("answer")
        if ans is None:
            return False
        if isinstance(ans, str) and ans.strip().lower() in EMPTY_ANSWER_LABELS:
            return False
        if isinstance(ans, str) and self.rule_engine.walk_in_not_customer and ans.strip().lower() in WALK_IN_CUSTOMER_LABELS:
            return False
        if isinstance(ans, (list, tuple)) and len(ans) == 0:
            return False
        return True

    @staticmethod
    def _apply_top_k(pred: Any, k: Optional[int]) -> Any:
        """Cắt list/table về k phần tử đầu khi câu hỏi hỏi top k (LLM đôi khi quên LIMIT)."""
        if not k or not isinstance(pred, dict):
            return pred
        ans = pred.get("answer")
        if pred.get("type") == "list" and isinstance(ans, list) and len(ans) > k:
            return dict(pred, answer=ans[:k])
        if pred.get("type") == "table" and isinstance(ans, dict) and isinstance(ans.get("rows"), list) and len(ans["rows"]) > k:
            return dict(pred, answer=dict(ans, rows=ans["rows"][:k]))
        return pred

    @staticmethod
    def _majority_vote(candidates: List[Tuple[str, Dict[str, Any]]]) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        """
        Gom các ứng viên hợp lệ theo đáp án tương đương (check_consensus) và chọn nhóm có >= 2 phiếu.
        Hai nhóm cùng số phiếu cao nhất -> không chọn (None).
        """
        groups: List[List[Tuple[str, Dict[str, Any]]]] = []
        for name, pred in candidates:
            for g in groups:
                if check_consensus(g[0][1], pred):
                    g.append((name, pred))
                    break
            else:
                groups.append([(name, pred)])
        if not groups:
            return None
        groups.sort(key=len, reverse=True)
        if len(groups[0]) < 2 or (len(groups) > 1 and len(groups[1]) == len(groups[0])):
            return None
        return groups[0][0][1], [n for n, _ in groups[0]]

    def _resolve_disagreement(
        self,
        qid: str,
        question: str,
        as_of_date: str,
        expected_type: str,
        pred1: Dict[str, Any],
        pred2: Dict[str, Any],
        raw1: Any,
        raw2: Any,
        code1: str,
        code2: str,
        rules_text: str = "",
    ) -> Tuple[Dict[str, Any], str, bool]:
        """Trọng tài theo đặc trưng truy vấn khi P1 và P2 không đồng thuận (và lá phiếu thứ ba không phân xử được)."""
        p1_valid = self._is_meaningful_pred(pred1)
        p2_valid = self._is_meaningful_pred(pred2)
        abstain = {"id": qid, "type": "abstain"}
        p1_code_str = str(code1 or "")
        p2_code_str = str(code2 or "")
        q_low = question.lower()

        # Trọng tài Khách lẻ: chỉ khi rulebook đang dùng ghi "khách lẻ không phải khách"
        def _raw_ans(raw: Any) -> str:
            val = raw.get("answer", "") if isinstance(raw, dict) else raw
            return unicodedata.normalize("NFC", str(val if val is not None else "")).strip().lower()

        walk_in_rule = self.rule_engine.walk_in_not_customer
        p1_is_generic = walk_in_rule and _raw_ans(raw1) in WALK_IN_CUSTOMER_LABELS
        p2_is_generic = walk_in_rule and _raw_ans(raw2) in WALK_IN_CUSTOMER_LABELS
        if p2_is_generic and p1_valid and pred1.get("type") == "string" and not p1_is_generic:
            return pred1, "retail_customer_arbitration_p1", False
        if p1_is_generic and p2_valid and pred2.get("type") == "string" and not p2_is_generic:
            return pred2, "retail_customer_arbitration_p2", False

        if p1_valid and p2_valid:
            # Câu hỏi số đơn: ưu tiên bên dùng COUNT(DISTINCT ...) (đếm dòng chi tiết thay vì đếm đơn là lỗi SQL chung)
            is_order_cnt_q = any(k in q_low for k in ["mấy đơn", "nhiêu đơn", "bao nhiêu đơn", "bn đơn", "số đơn", "mấy hóa đơn", "mấy phiếu"])
            p1_has_distinct = "COUNT(DISTINCT" in p1_code_str.upper()
            p2_has_distinct = "COUNT(DISTINCT" in p2_code_str.upper()
            if pred1.get("type") == "number" and is_order_cnt_q and p1_has_distinct and not p2_has_distinct:
                return pred1, "p1_distinct_semantic_arbitration", False
            if pred2.get("type") == "number" and is_order_cnt_q and p2_has_distinct and not p1_has_distinct:
                return pred2, "p2_distinct_semantic_arbitration", False
            if pred1.get("type") == "number" and pred2.get("type") == "number" and self.arbitration_strategy in ("tournament", "adaptive"):
                referee_choice = self.arbitrate_safe_tournament(
                    question=question,
                    as_of_date=as_of_date,
                    code1=p1_code_str,
                    ans1=pred1.get("answer"),
                    code2=p2_code_str,
                    ans2=pred2.get("answer"),
                    rules_text=rules_text,
                )
                if referee_choice == 1:
                    return pred1, "tournament_referee_p1", False
                if referee_choice == 2:
                    return pred2, "tournament_referee_p2", False
            if expected_type == "table" and pred1.get("type") == "table" and pred2.get("type") == "table":
                return abstain, "table_disagreement_abstain", False
            return abstain, "consensus_disagreement_abstain", False

        if p1_valid and not p2_valid:
            if self.arbitration_strategy in ("tournament", "adaptive"):
                # Prompt 1 vượt qua toàn bộ guard an toàn, Prompt 2 lỗi kỹ thuật -> Giữ Prompt 1
                return pred1, "p1_valid_p2_failed_fallback", False
            return abstain, "strict_requires_both", False

        if p2_valid and not p1_valid:
            if self.arbitration_strategy in ("tournament", "adaptive"):
                return pred2, "p2_valid_p1_failed_fallback", False
            return abstain, "p2_only_rejected_abstain", False

        return abstain, "both_failed", False

    def answer_question_dual(
        self, question_obj: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Giải quyết câu hỏi bằng kiến trúc Dual-Prompt Consensus Gate & Tiered Arbitration:
          - Prompt 1: SQL-Centric trên DuckDB (seed).
          - Prompt 2: Defensive CTE trên DuckDB (seed + 58).
          - Đồng thuận P1 == P2 -> nhận đáp án.
          - Bất đồng / một bên lỗi -> Prompt 3 (plan-then-code) làm lá phiếu thứ ba; đáp án có >= 2 phiếu được nhận.
          - Vẫn không phân xử được -> trọng tài theo đặc trưng truy vấn, mặc định Safe Abstain (0 điểm).
        """
        qid = question_obj["id"]
        business_id = question_obj["business_id"]
        as_of_date = question_obj["as_of_date"]
        raw_question = question_obj.get("question", "")
        question = raw_question

        try:
            biz_readme = self.store.get_readme(business_id)
            tables = self.store.load_tables(business_id)
            schema_prompt = self.store.build_compact_schema_prompt(business_id, question=raw_question)

            # Tầng 1a: chuẩn hoá tất định (Unicode NFC, viết tắt) -> dùng cho mọi bước tất định
            question_det = self.rule_engine.normalize_telex_typos(raw_question) or raw_question
            # Giải mã chữ gõ Telex thô ("thangs" -> "tháng") theo luật gõ, bảo vệ từ thuộc dữ liệu của shop
            question_det = decode_telex_text(question_det, self._protected_tokens(business_id, tables))
            expected_type = self.rule_engine.infer_expected_type(question_det)
            rules_text = self.rule_engine.select_relevant_rules(question_det, biz_readme)

            # Kiểm tra "không trả lời được" tất định TRƯỚC mọi lời gọi LLM
            is_unans, unans_reason = self.rule_engine.detect_unanswerable_deterministic(
                question_obj=dict(question_obj, question=question_det),
                store=self.store,
                date_resolver=self.date_resolver,
                rulebook_text=self.rule_engine.full_rulebook,
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

            # Tầng 1b (tuỳ chọn): viết lại câu hỏi theo ngữ cảnh DB (DART-SQL) -> chỉ dùng làm gợi ý cho LLM
            question = question_det
            if self.rewrite_question:
                question = self.vietnamese_normalizer.canonicalize(
                    question=raw_question,
                    schema_summary=schema_prompt,
                    as_of_date=as_of_date,
                    tables=tables,
                ) or question_det
                if expected_type == "number" and question != question_det:
                    expected_type = self.rule_engine.infer_expected_type(question)

            hint_question = question_det if question == question_det else f"{question_det} || {question}"
            entity_hints = self.entity_grounder.find_grounding_hints(hint_question, business_id, tables)
            value_hints = load_lexicon().build_value_hints(question_det, tables)
            if value_hints:
                entity_hints = (entity_hints + "\n\n" + value_hints) if entity_hints else value_hints
            few_shot_prompt = self.few_shot_selector.select_exemplars(
                question=question_det,
                has_entity_hint=bool(entity_hints and entity_hints.strip()),
            )
            date_question = question_det
            date_ctx = self.date_resolver.resolve(question_det, as_of_date)
            if not date_ctx.get("detected_periods") and question != question_det:
                alt_ctx = self.date_resolver.resolve(question, as_of_date)
                if alt_ctx.get("detected_periods"):
                    date_ctx, date_question = alt_ctx, question
            date_prompt = self.date_resolver.format_for_prompt(date_question, as_of_date)
            # Chỉ nêu TÊN các mục liên quan; nội dung rulebook đầy đủ đã nằm trong system prompt dùng chung
            rules_prompt = "\n".join(
                f"- {ln.lstrip('#').strip()}" for ln in rules_text.splitlines() if ln.startswith("#")
            )
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
            question_obj=dict(question_obj, question=question),
            expected_type=expected_type,
            date_prompt=date_prompt,
            rules_prompt=rules_prompt,
            schema_prompt=schema_prompt,
            entity_hints=entity_hints,
            few_shot_prompt=few_shot_prompt,
            original_question=raw_question,
        )
        top_k = self._extract_top_k(question_det)

        def _run(system_prompt: str, seed: int) -> Tuple[Dict[str, Any], str, Any]:
            pred, code, raw = self._solve_single_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                qid=qid,
                business_id=business_id,
                as_of_date=as_of_date,
                date_ctx=date_ctx,
                expected_type=expected_type,
                question=question_det,
                seed=seed,
            )
            return self._apply_top_k(pred, top_k), code, raw

        # Nếu cấu hình single prompt mode (chỉ Prompt 1)
        if self.consensus_mode == "single":
            pred1, code1, raw1 = _run(SYSTEM_PROMPT_SQL, self.seed)
            p1_valid = self._is_meaningful_pred(pred1)
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
            fut1 = executor.submit(_run, SYSTEM_PROMPT_SQL, self.seed)
            fut2 = executor.submit(_run, SYSTEM_PROMPT_DEFENSIVE_SQL, self.seed + 58)
            pred1, code1, raw1 = fut1.result()
            pred2, code2, raw2 = fut2.result()

        p1_valid = self._is_meaningful_pred(pred1)
        p2_valid = self._is_meaningful_pred(pred2)
        pred3: Optional[Dict[str, Any]] = None
        code3, raw3 = "", None

        if p1_valid and p2_valid and check_consensus(pred1, pred2):
            final_pred, decision_reason, is_consensus = pred1, "consensus_match", True
        else:
            final_pred, decision_reason, is_consensus = {"id": qid, "type": "abstain"}, "", False
            if self.tiebreak and (p1_valid or p2_valid):
                # Lá phiếu thứ ba: Prompt 3 (plan-then-code). Chỉ nhận đáp án được >= 2/3 ứng viên ủng hộ.
                pred3, code3, raw3 = _run(SYSTEM_PROMPT_PLAN_SQL, self.seed + 17)
                cands = [
                    (name, p) for name, p in (("p1", pred1), ("p2", pred2), ("p3", pred3))
                    if self._is_meaningful_pred(p)
                ]
                voted = self._majority_vote(cands)
                if voted is not None:
                    final_pred = voted[0]
                    decision_reason = "majority_vote_" + "_".join(voted[1])
                    is_consensus = True
            if not decision_reason:
                final_pred, decision_reason, is_consensus = self._resolve_disagreement(
                    qid=qid,
                    question=question_det,
                    as_of_date=as_of_date,
                    expected_type=expected_type,
                    pred1=pred1,
                    pred2=pred2,
                    raw1=raw1,
                    raw2=raw2,
                    code1=code1,
                    code2=code2,
                    rules_text=rules_text,
                )

        if final_pred.get("type") != "abstain":
            final_pred = dict(final_pred, id=qid)

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
            "prompt3_code": code3,
            "prompt3_raw": raw3,
            "prompt3_pred": pred3,
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
