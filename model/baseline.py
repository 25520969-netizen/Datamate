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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

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
   - Tất cả tên bảng và tên cột tiếng Việt có dấu, khoảng trắng BẮT BUỘC bọc trong ngoặc kép: `"Thành tiền"`, `"Mã đơn hàng"`, `"Ngày"`, `"Trạng thái"`.
   - Lọc ngày: `WHERE CAST("Ngày" AS DATE) BETWEEN DATE 'start' AND DATE 'end'` (hoặc `CAST("Ngày" AS VARCHAR)`).
   - Lọc đơn hợp lệ (bỏ đơn hủy, đơn trả): `WHERE "status_norm" = 'Hoàn thành'` (hoặc `"Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%trả%'`).
   - Xếp hạng / Bán chạy: `ORDER BY SUM(COALESCE("qty_base", "Số lượng")) DESC LIMIT N`. Tuyệt đối không xếp hạng bán chạy theo doanh thu!
   - Đếm đơn hàng: `SELECT COUNT(DISTINCT "Mã đơn hàng")` trên bảng chi tiết; không dùng `COUNT(*)` kẻo đếm nhầm dòng sản phẩm.
   - Xếp hạng khách hàng: Bỏ khách lẻ (`WHERE "Khách hàng" NOT ILIKE '%khách lẻ%' AND "Khách hàng" IS NOT NULL`).
2. Định dạng biến kết quả cuối cùng:
   - `number`: `result = {"type": "number", "answer": float(val)}`
   - `percent`: `result = {"type": "percent", "answer": float(val)}` (đã nhân 100, ví dụ: 15.5)
   - `date`: `result = {"type": "date", "answer": "YYYY-MM-DD"}`
   - `string`: `result = {"type": "string", "answer": "Tên đối tượng"}`
   - `list`: `result = {"type": "list", "answer": [item1, item2, ...]}`
   - `table`: `result = {"columns": ["khach_hang", "so_tien"], "rows": [["Tên A", 1000], ...]}`
   - `unanswerable`: `result = {"type": "unanswerable", "reason": "..."}`
3. Chỉ trả về một khối mã ```python ... ``` duy nhất, không giải thích văn bản thừa."""

# -----------------------------------------------------------------------------
# PROMPT 2: Tư duy Thủ tục từng bước (Defensive Procedural / Step-by-Step Python)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_DEFENSIVE = """Bạn là Chuyên gia Khoa học Dữ liệu chú trọng tính chính xác và kiểm soát rủi ro (DataMate 2026).
Nhiệm vụ: Viết mã Python xử lý tuần tự trên biến `tables` (dict DataFrames) hoặc `con` DuckDB, đặc biệt chú ý các điều kiện biên và quy tắc loại trừ.

QUY TẮC CỐT LÕI (PHÒNG THỦ TỪNG BƯỚC):
1. Thực hiện xử lý phòng thủ theo từng bước rõ ràng:
   - Bước 1 (Lọc an toàn): Loại bỏ ngay các dòng trạng thái 'Đã hủy', 'Trả hàng', hoặc các dòng tổng cộng/subtotal.
   - Bước 2 (Lọc thời gian): Lọc đúng khoảng ngày yêu cầu trên cột ngày hoặc mốc thời gian đã cho (`date_context`).
   - Bước 3 (Ghép nối & Phân nhóm): Ghép bảng theo khóa ngoại hoặc nhóm (`groupby`) theo đúng tiêu chí câu hỏi.
   - Bước 4 (Tính toán): Tính tổng tiền, số lượng, hoặc tỷ lệ phần trăm (đã nhân 100).
2. Kiểm tra an toàn: Nếu sau khi lọc dữ liệu rỗng, gán giá trị mặc định 0 hoặc danh sách rỗng, tránh lỗi crash.
3. Định dạng biến kết quả cuối cùng:
   - `number`: `result = {"type": "number", "answer": float(val)}`
   - `percent`: `result = {"type": "percent", "answer": float(val)}` (đã nhân 100, ví dụ: 40.3)
   - `date`: `result = {"type": "date", "answer": "YYYY-MM-DD"}`
   - `string`: `result = {"type": "string", "answer": "Tên đối tượng"}`
   - `list`: `result = {"type": "list", "answer": [item1, item2, ...]}`
   - `table`: `result = {"columns": ["cột 1", "cột 2"], "rows": [[val1, val2], ...]}`
   - `unanswerable`: `result = {"type": "unanswerable", "reason": "..."}`
4. Chỉ trả về một khối mã ```python ... ``` duy nhất, không giải thích văn bản thừa."""

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
        max_tokens: int = 640,
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

        self._hf_tokenizer: Any = None
        self._hf_model: Any = None

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
            url = f"{self.api_base}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "seed": seed,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        if self.backend == "ollama":
            url = f"{self.api_base}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": seed,
                    "num_predict": self.max_tokens,
                },
            }
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
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
        )
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.verbose = verbose

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
            f"=== CÂU HỎI CẦN GIẢI QUYẾT ===\n"
            f"- Mã câu hỏi: {question_obj.get('id', 'unknown')}\n"
            f"- Cửa hàng (`business_id`): {business_id}\n"
            f"- Ngày hiện tại (`as_of_date` - Hôm nay): {as_of_date}\n"
            f"- Câu hỏi: {question}\n"
            f"- Kiểu đáp án kỳ vọng (`expected_type`): {expected_type}",
            f"=== NGỮ CẢNH THỜI GIAN ĐÃ QUY ĐỔI ===\n{date_prompt}",
            schema_prompt,
        ]
        if entity_hints:
            parts.append(entity_hints)
        if few_shot_prompt:
            parts.append(few_shot_prompt)
        parts.extend([
            f"=== LUẬT NGHIỆP VỤ LIÊN QUAN ===\n{rules_prompt}",
            "=== YÊU CẦU SINH MÃ PYTHON ===\n"
            "Hãy viết mã Python truy vấn dữ liệu từ `tables` hoặc `con` DuckDB để tính ra kết quả chính xác.\n"
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
        total_attempts = self.max_retries + 1

        for attempt in range(total_attempts):
            try:
                llm_reply = self.llm_client.generate(messages, seed=seed)
                last_code = llm_reply
            except Exception as llm_exc:
                if self.verbose:
                    print(f"    ⚠️ [LLM Error] {qid} ({business_id}, attempt {attempt+1}): {llm_exc}", flush=True)
                return {"id": qid, "type": "abstain"}, last_code, None

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

        return {"id": qid, "type": "abstain"}, last_code, last_raw

    def answer_question_dual(
        self, question_obj: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Giải quyết câu hỏi bằng kiến trúc Dual-Prompt Consensus Gate:
          - Prompt 1: SQL-Centric trên DuckDB (Seed 42).
          - Prompt 2: Defensive Step-by-Step trên Pandas (Seed 100).
          - Chỉ nhận đáp án khi cả hai lời giải đồng thuận giá trị chuẩn.
        Trả về: (final_pred, details_dict) để xuất đồng thời 2 file JSONL.
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

        # 1. Chạy Prompt 1: SQL-Centric
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

        # 2. Chạy Prompt 2: Defensive Step-by-Step
        pred2, code2, raw2 = self._solve_single_prompt(
            system_prompt=SYSTEM_PROMPT_DEFENSIVE,
            user_prompt=user_prompt,
            qid=qid,
            business_id=business_id,
            as_of_date=as_of_date,
            date_ctx=date_ctx,
            expected_type=expected_type,
            question=question,
            seed=100,
        )

        # 3. So khớp đồng thuận (Consensus Gate)
        is_consensus = check_consensus(pred1, pred2)
        if is_consensus:
            final_pred = pred1
        else:
            final_pred = {"id": qid, "type": "abstain"}

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
            "final_decision": final_pred.get("type"),
            "final_answer": final_pred.get("answer"),
        }
        return final_pred, details

    def answer_question(self, question_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Tương thích ngược với interface chuẩn."""
        final_pred, _ = self.answer_question_dual(question_obj)
        return final_pred
