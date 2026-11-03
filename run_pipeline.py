#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy toàn bộ pipeline Multi-Stage Code-Generation Agent (DataMate)
với mô hình mặc định `Qwen/Qwen2.5-Coder-7B-Instruct`.

Ví dụ sử dụng:
  1. Chạy thử nghiệm trên 20 câu đầu của tập train (kèm tự động chấm điểm PA):
     .venv/bin/python run_pipeline.py --limit 20 --evaluate

  2. Chạy với Ollama local (`qwen2.5-coder:7b`):
     .venv/bin/python run_pipeline.py --backend ollama --model qwen2.5-coder:7b --limit 50 --evaluate

  3. Chạy với vLLM / OpenAI-compatible server (`Qwen/Qwen2.5-Coder-7B-Instruct`):
     .venv/bin/python run_pipeline.py --backend openai --api-base http://localhost:8000/v1 --evaluate

  4. Chạy offline trực tiếp qua HuggingFace Transformers trên GPU:
     .venv/bin/python run_pipeline.py --backend transformers --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluate import evaluate_predictions, load_jsonl
from model.baseline import DEFAULT_MODEL, DEFAULT_QWEN_14B_AWQ_MODEL, DataMateAgent, QwenCoderClient


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataMate Multi-Stage Code-Generation Pipeline (Qwen2.5-Coder-7B-Instruct)"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "train" / "questions.jsonl",
        help="Đường dẫn tới file questions.jsonl đầu vào.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "train" / "data",
        help="Đường dẫn tới thư mục chứa dữ liệu các shop (biz_*).",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=None,
        help="File calendar.csv. Mặc định tự tìm cạnh dữ liệu: <data-root>, thư mục cha của data-root, thư mục chứa file câu hỏi.",
    )
    parser.add_argument(
        "--rulebook",
        type=Path,
        default=None,
        help="File rulebook.md của TẬP đang chạy. Mặc định tự tìm cạnh dữ liệu (như --calendar). "
             "Không tìm thấy thì dừng báo lỗi, trừ khi bật --allow-default-rulebook.",
    )
    parser.add_argument(
        "--allow-default-rulebook",
        action="store_true",
        help="Cho phép dùng assets/rulebook.md (rulebook bộ train) khi không tìm thấy rulebook của tập. Chỉ dùng khi thử nghiệm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submission.jsonl",
        help="Đường dẫn xuất file kết quả submission.jsonl.",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Đường dẫn xuất file chi tiết so sánh 2 prompt (mặc định: <output>_details.jsonl).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "ollama", "transformers", "mock"],
        default="openai",
        help="Backend thực thi LLM (mặc định: openai / vLLM, hỗ trợ 'mock' cho CI/Dry-run).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Tên hoặc đường dẫn mô hình (mặc định: Qwen/Qwen2.5-Coder-14B-Instruct-AWQ).",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="URL endpoint cho backend openai hoặc ollama.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Số lần tối đa tự sửa lỗi qua Sandbox Traceback (mặc định: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Giới hạn thời gian thực thi mỗi đoạn code trong Sandbox (giây).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số lượng câu hỏi cần chạy (0 = chạy toàn bộ).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Tự động đối chiếu với answers.jsonl sau khi chạy xong.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In chi tiết từng lần thử sinh code và kết quả Sandbox.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Số câu hỏi xử lý song song (mỗi câu gửi 2 request P1/P2 cùng lúc -> 8 sequence đồng thời, vừa KV cache "
             "của 14B-AWQ trên GPU ~22GB nhờ system prompt dùng chung được prefix-cache). Mặc định: 4.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "train" / "answers.jsonl",
        help="Đường dẫn tới file answers.jsonl chuẩn (dùng khi có --evaluate).",
    )
    parser.add_argument(
        "--consensus-mode",
        type=str,
        default="dual",
        choices=["dual", "single"],
        help="Chế độ sinh code: 'dual' (2 prompt so khớp) hoặc 'single' (chỉ Prompt 1 SQL).",
    )
    parser.add_argument(
        "--arbitration",
        type=str,
        default="strict",
        choices=["strict", "adaptive", "tournament"],
        help="Chính sách trọng tài: 'strict' (an toàn tối đa, yêu cầu đồng thuận tuyệt đối, Safe Abstain khi bất đồng), 'adaptive' (cho phép fallback P1 khi P2 lỗi runtime), hoặc 'tournament' (thêm trọng tài LLM khi hai số bất đồng).",
    )
    parser.add_argument(
        "--rewrite-question",
        action="store_true",
        help="Bật bước viết lại câu hỏi bằng LLM (DART-SQL) trước khi sinh code. Tốn thêm một lượt LLM tuần tự mỗi câu; mặc định tắt.",
    )
    parser.add_argument(
        "--detail-json",
        type=Path,
        default=None,
        help="Ghi file chi tiết dạng JSON (mảng) mô tả quyết định của Prompt 1, Prompt 2, Prompt 3 và lý do chốt đáp án; "
             "có đối chiếu đáp án chuẩn nếu có --gold.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Ghi báo cáo chấm điểm (đúng/sai/abstain, PA, theo từng kiểu) dạng JSON khi có --evaluate.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="In một dòng tiến độ sau mỗi N câu hoàn thành (mặc định 5).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Thư mục ghi cache Parquet của dữ liệu đã nạp (mặc định: .cache/ingested cạnh thư mục tập). Nên đặt ở nơi có quyền ghi.",
    )
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=4,
        help="Số tiến trình nạp dữ liệu song song cho các shop chưa có cache (mặc định 4; 1 = tuần tự).",
    )
    parser.add_argument(
        "--no-tiebreak",
        action="store_true",
        help="Tắt lá phiếu thứ ba (Prompt 3) khi P1/P2 bất đồng; mặc định bật và chỉ nhận đáp án có >= 2/3 phiếu.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tái sử dụng các đáp án không-abstain trong file output cũ (chỉ dùng khi tiếp tục một lần chạy bị ngắt CÙNG phiên bản code).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="(Giữ để tương thích) Chạy mới toàn bộ câu hỏi; đây đã là mặc định nếu không có --resume.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Cố định random seed cho toàn bộ pipeline và mô hình LLM (mặc định: 42).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Thời gian chờ tối đa cho mỗi request tới LLM API (giây, mặc định: 300.0).",
    )
    return parser.parse_args()


def _ingest_worker(data_root: str, cache_root: str, business_id: str) -> str:
    """Chạy trong tiến trình con: nạp một shop và ghi Parquet cache (tiến trình chính đọc lại từ cache)."""
    from src.ingest import BusinessDataStore

    BusinessDataStore(data_root=Path(data_root), cache_root=Path(cache_root)).load_tables(business_id, use_cache=True)
    return business_id


def _parallel_ingest(store: Any, biz_ids: List[str], workers: int) -> None:
    """
    Làm sạch/nạp lần đầu mỗi shop tốn 1-60 giây (CPU). Chạy các shop chưa có cache song song bằng nhiều tiến trình,
    ghi Parquet cache; vòng nạp tuần tự phía sau chỉ còn đọc cache. Lỗi ở tiến trình con không ảnh hưởng: tiến trình
    chính tự nạp lại như cũ.
    """
    cache_root = getattr(store, "cache_root", None)
    if cache_root is None:
        return
    pending = [b for b in biz_ids if not (cache_root / b / "_manifest.json").exists()]
    workers = min(workers, len(pending), os.cpu_count() or 1)
    if workers <= 1:
        return
    print(f"⚙️ Nạp song song {len(pending)} shop chưa có cache bằng {workers} tiến trình...", flush=True)
    from concurrent.futures import ProcessPoolExecutor

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_ingest_worker, str(store.data_root), str(store.cache_root), b) for b in pending]
            for f in futs:
                try:
                    f.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"   ⚠️ nạp song song lỗi ({type(exc).__name__}), sẽ nạp tuần tự: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Không chạy được nạp song song: {exc}", flush=True)


def resolve_split_files(
    data_root: Path,
    questions: Path,
    rulebook: Optional[Path] = None,
    calendar: Optional[Path] = None,
    allow_default_rulebook: bool = False,
) -> Tuple[Path, Path]:
    """
    Rulebook và lịch âm dương là của TỪNG TẬP (split). Thứ tự tìm: đường dẫn chỉ định -> data_root ->
    thư mục cha của data_root -> thư mục chứa file câu hỏi. Rulebook không tìm thấy thì dừng báo lỗi
    (trừ khi allow_default_rulebook); calendar không thấy thì dùng assets/calendar.csv (lịch giống nhau giữa các tập).
    """
    def _find(name: str, explicit: Optional[Path]) -> Optional[Path]:
        if explicit is not None:
            if explicit.exists():
                return explicit
            raise SystemExit(f"❌ Không tìm thấy {name} được chỉ định: {explicit}")
        for base in (data_root, data_root.parent, questions.parent):
            cand = base / name
            if cand.exists():
                return cand
        return None

    found_rb = _find("rulebook.md", rulebook)
    if found_rb is None:
        default_rb = PROJECT_ROOT / "assets" / "rulebook.md"
        if not (allow_default_rulebook and default_rb.exists()):
            raise SystemExit(
                "❌ Không tìm thấy rulebook.md của tập dữ liệu (đã tìm trong data-root, thư mục cha và thư mục file câu hỏi). "
                "Hãy truyền --rulebook, hoặc dùng --allow-default-rulebook để dùng bản train (chỉ khi thử nghiệm)."
            )
        print("⚠️ CẢNH BÁO: dùng assets/rulebook.md (rulebook bộ train), KHÔNG phải rulebook của tập đang chạy!", flush=True)
        found_rb = default_rb

    found_cal = _find("calendar.csv", calendar)
    if found_cal is None:
        default_cal = PROJECT_ROOT / "assets" / "calendar.csv"
        if not default_cal.exists():
            raise SystemExit("❌ Không tìm thấy calendar.csv (data-root, thư mục cha, thư mục file câu hỏi hoặc assets/).")
        print("⚠️ Không thấy calendar.csv cạnh dữ liệu, dùng assets/calendar.csv", flush=True)
        found_cal = default_cal
    return found_rb, found_cal


def main() -> None:
    import random
    import numpy as np
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    args = parse_args()

    # Cố định random seed toàn cục để đảm bảo 100% tính tái lập kết quả
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    # Sửa lỗi đường dẫn thừa '/data': chỉ khi thư mục cha chứa các thư mục shop (có README.md), không dò theo tiền tố id shop.
    # Không tự đổi --data-root sang thư mục "preprocessed" nào: luôn chạy trên đúng dữ liệu được chỉ định.
    if not args.data_root.exists() and args.data_root.name == "data" and args.data_root.parent.exists():
        if any((d / "README.md").exists() for d in args.data_root.parent.iterdir() if d.is_dir()):
            print(f"ℹ️ [Path Correction] Không có '{args.data_root}', dùng thư mục cha: {args.data_root.parent}")
            args.data_root = args.data_root.parent

    # Tự động định vị file questions nếu đường dẫn tương đối không nằm tại cwd
    if not args.questions.exists():
        if (PROJECT_ROOT / args.questions).exists():
            args.questions = PROJECT_ROOT / args.questions
        elif (PROJECT_ROOT / args.questions.name).exists():
            args.questions = PROJECT_ROOT / args.questions.name

    # Tương tự tự động định vị file gold
    if args.gold and not args.gold.exists():
        if (PROJECT_ROOT / args.gold).exists():
            args.gold = PROJECT_ROOT / args.gold
        elif (PROJECT_ROOT / args.gold.name).exists():
            args.gold = PROJECT_ROOT / args.gold.name

    args.rulebook, args.calendar = resolve_split_files(
        data_root=args.data_root,
        questions=args.questions,
        rulebook=args.rulebook,
        calendar=args.calendar,
        allow_default_rulebook=args.allow_default_rulebook,
    )
    _rb_title = args.rulebook.read_text(encoding="utf-8").splitlines()[0] if args.rulebook.stat().st_size else ""
    print(f"📘 Rulebook: {args.rulebook} | {_rb_title}", flush=True)
    print(f"📅 Calendar: {args.calendar}", flush=True)

    questions = load_jsonl(args.questions)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    if args.backend == "mock":
        def mock_generator(messages: List[Dict[str, str]]) -> str:
            # Deterministic mock code cho dry-run test
            return '```python\nresult = {"type": "number", "answer": 123456}\n```'

        llm_client = QwenCoderClient(
            backend="custom",
            model_name=args.model,
            custom_generator=mock_generator,
            request_timeout=args.request_timeout,
        )
    else:
        llm_client = QwenCoderClient(
            backend=args.backend,
            model_name=args.model,
            api_base=args.api_base,
            request_timeout=args.request_timeout,
        )

    print(f"Kiểm tra kết nối tới LLM endpoint ({llm_client.backend} @ {llm_client.api_base})...")
    if not llm_client.health_check(timeout=5.0):
        print(f"⚠️ CẢNH BÁO: Chưa kết nối được tới LLM endpoint tại {llm_client.api_base}!")
        print(f"   Nếu đang dùng vLLM/Ollama, hãy đảm bảo server đã chạy xong bước khởi động.")
    else:
        print(f"✅ Kết nối LLM endpoint thành công! Sẵn sàng phục vụ.")

    agent = DataMateAgent(
        data_root=args.data_root,
        calendar_path=args.calendar,
        rulebook_path=args.rulebook,
        llm_client=llm_client,
        sandbox_timeout=args.timeout,
        max_retries=args.max_retries,
        verbose=args.verbose,
        consensus_mode=args.consensus_mode,
        arbitration_strategy=args.arbitration,
        cache_root=args.cache_root,
        seed=args.seed,
        tiebreak=not args.no_tiebreak,
        rewrite_question=args.rewrite_question,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    details_path = args.details_output or args.output.with_name(args.output.stem + "_details.jsonl")

    # Nạp các câu trả lời tốt đã có từ trước (không phải abstain) nếu không bật cờ --fresh
    existing_cache: Dict[str, Dict[str, Any]] = {}
    existing_details: Dict[str, Dict[str, Any]] = {}
    reuse_previous = args.resume and not args.fresh
    if reuse_previous and args.output.exists():
        for ln in args.output.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    item = json.loads(ln)
                    if item.get("type") != "abstain":
                        existing_cache[item["id"]] = item
                except Exception:
                    pass
    if reuse_previous and details_path.exists():
        for ln in details_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    d_item = json.loads(ln)
                    if "id" in d_item:
                        existing_details[d_item["id"]] = d_item
                except Exception:
                    pass
    if existing_cache:
        print(f"✅ Đã bảo lưu {len(existing_cache)} câu trả lời tốt từ trước (bỏ qua không cần tính lại).")

    # Chuẩn bị và nạp trước Parquet cache cho các doanh nghiệp cần truy vấn để các luồng không tranh chấp IO
    biz_ids = sorted(list({q["business_id"] for q in questions}))
    print(f"Kiểm tra và nạp sẵn dữ liệu cho {len(biz_ids)} doanh nghiệp...")
    t_prewarm = time.time()
    _parallel_ingest(agent.store, biz_ids, args.ingest_workers)
    for idx, bid in enumerate(biz_ids, 1):
        t0 = time.time()
        agent.store.load_tables(bid, use_cache=True)
        print(f"  [{idx:02d}/{len(biz_ids)}] {bid}: Sẵn sàng ({time.time() - t0:.1f}s)", flush=True)
    print(f"✅ Đã chuẩn bị xong dữ liệu {len(biz_ids)} doanh nghiệp trong {time.time() - t_prewarm:.1f}s!")

    results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    details_results: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    completed_count = 0

    for idx, q_obj in enumerate(questions):
        qid = q_obj["id"]
        if qid in existing_cache:
            results[idx] = existing_cache[qid]
            if qid in existing_details:
                details_results[idx] = existing_details[qid]
            completed_count += 1

    to_run = [(idx, q_obj) for idx, q_obj in enumerate(questions) if results[idx] is None]
    # Sắp xếp câu hỏi theo business_id để kích hoạt tối đa vLLM Prefix Caching
    # (Toàn bộ câu hỏi cùng shop có schema và rules đầu prompt giống nhau 100%, tái sử dụng KV cache)
    # Thứ tự ghi ra results[idx] vẫn đảm bảo 100% khớp với file câu hỏi gốc.
    to_run.sort(key=lambda item: (item[1].get("business_id", ""), item[0]))
    print(f"Bắt đầu chạy DataMate Pipeline trên {len(questions)} câu hỏi (Cần chạy mới: {len(to_run)} | Concurrency: {args.concurrency})...")
    print(f"Model: {llm_client.model_name} | Backend: {llm_client.backend} | Max Retries: {args.max_retries}")

    lock = threading.Lock()
    start_time = time.time()

    # Ghi NGAY từng câu vào file đích (chế độ append + flush + fsync): nếu lượt chạy bị dừng giữa chừng
    # (hết giờ, bị kill), mọi câu đã xong vẫn nằm trong file nộp; câu chưa xong tính là abstain.
    out_f = args.output.open("w", encoding="utf-8")
    out_d = details_path.open("w", encoding="utf-8")

    def _write_line(fh: Any, obj: Dict[str, Any]) -> None:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass

    def _record(idx: int) -> None:
        if results[idx] is not None:
            _write_line(out_f, results[idx])
        if details_results[idx] is not None:
            _write_line(out_d, details_results[idx])

    for _i in range(len(questions)):
        _record(_i)

    def _safe_answer(q_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Một câu lỗi bất ngờ chỉ được abstain (0 điểm), không được làm sập cả lượt chạy."""
        try:
            return agent.answer_question_dual(q_obj)
        except Exception as exc:
            print(f"    ⚠️ [Unhandled] {q_obj.get('id')}: {type(exc).__name__}: {exc}", flush=True)
            return {"id": q_obj["id"], "type": "abstain"}, {
                "id": q_obj["id"],
                "business_id": q_obj.get("business_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "is_consensus": False,
                "decision_reason": "unhandled_exception",
                "final_decision": "abstain",
                "final_answer": None,
            }

    def _solve_item(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        idx, q_obj = item
        pred, details = _safe_answer(q_obj)
        return idx, pred, details

    gold_map: Dict[str, Dict[str, Any]] = {}
    if args.evaluate and args.gold and args.gold.exists():
        gold_map = {g["id"]: g for g in load_jsonl(args.gold)}
    tally = {"correct": 0, "wrong": 0, "abstain": 0}
    n_new = len(to_run)

    def _on_done(idx: int, pred: Dict[str, Any], details: Dict[str, Any]) -> None:
        """Ghi kết quả một câu và in một dòng log (kèm điểm PA tạm thời khi có đáp án chuẩn)."""
        nonlocal completed_count
        results[idx] = pred
        details_results[idx] = details
        completed_count += 1
        _record(idx)

        done_new = completed_count - (len(questions) - n_new)
        q_obj = questions[idx]
        verdict = ""
        if gold_map and q_obj["id"] in gold_map:
            from evaluate import score_single_prediction

            _, status = score_single_prediction(pred, gold_map[q_obj["id"]])
            tally[status] += 1
            verdict = {"correct": " ✔ĐÚNG", "wrong": " ✘SAI", "abstain": " ·bỏ qua"}[status]
        if done_new == 1 or done_new % max(args.log_every, 1) == 0 or completed_count == len(questions):
            elapsed = time.time() - start_time
            avg_speed = elapsed / max(done_new, 1)
            rem_mins = (n_new - done_new) * avg_speed / 60.0
            ans_preview = json.dumps(pred.get("answer"), ensure_ascii=False) if "answer" in pred else str(pred.get("type"))
            if len(ans_preview) > 35:
                ans_preview = ans_preview[:32] + "..."
            if details.get("is_consensus"):
                status_tag = "ĐỒNG THUẬN"
            elif pred.get("type") != "abstain":
                status_tag = f"QUYẾT ĐỊNH ({details.get('decision_reason', '')})"
            else:
                status_tag = f"ABSTAIN ({details.get('decision_reason', '')})"
            line = (
                f"[{completed_count}/{len(questions)}] ({completed_count / len(questions) * 100:.1f}%) [{status_tag}] "
                f"{q_obj['id']} ({q_obj['business_id']}) -> {pred.get('type')}: {ans_preview}{verdict} | "
                f"{avg_speed:.2f}s/câu | ETA {rem_mins:.1f}m"
            )
            if gold_map:
                n = sum(tally.values())
                pa = (tally["correct"] - tally["wrong"]) / n * 100 if n else 0.0
                line += f" | PA tạm {pa:.1f}% (đúng {tally['correct']}/sai {tally['wrong']}/bỏ qua {tally['abstain']})"
            print(f"  {line}", flush=True)
            if pred.get("type") == "abstain":
                for n in (1, 2):
                    p_pred = details.get(f"prompt{n}_pred") or {}
                    err = p_pred.get("error") if isinstance(p_pred, dict) else None
                    if err:
                        print(f"      ↳ P{n}: {' '.join(str(err).split())[-220:]}", flush=True)

    if to_run:
        if args.concurrency > 1 and len(to_run) > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_solve_item, item): item for item in to_run}
                for fut in as_completed(futures):
                    idx, pred, details = fut.result()
                    with lock:
                        _on_done(idx, pred, details)
        else:
            for idx, q_obj in to_run:
                pred, details = _safe_answer(q_obj)
                _on_done(idx, pred, details)

    out_f.close()
    out_d.close()
    # Ghi lại theo đúng thứ tự file câu hỏi, thay thế nguyên tử (không bao giờ để file nộp ở trạng thái dở dang)
    for path, rows in ((args.output, results), (details_path, details_results)):
        tmp_path = path.with_name(path.name + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                if r is not None:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)

    predictions = [r for r in results if r is not None]
    print(f"✅ Đã xuất kết quả nộp bài (jsonl): {args.output}")
    print(f"✅ Đã xuất log chi tiết (jsonl): {details_path}")

    report = None
    if args.evaluate and args.gold and args.gold.exists():
        all_gold = load_jsonl(args.gold)
        q_ids = {q["id"] for q in questions}
        subset_gold = [g for g in all_gold if g["id"] in q_ids]
        report = evaluate_predictions(predictions, subset_gold)
        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP CÂU HỎI ĐÃ CHẠY")
        print("=" * 60)
        print(f"Tổng số câu (N)          : {report['total']}")
        print(f"Đúng (+1)                : {report['correct']}")
        print(f"Sai (-1)                 : {report['wrong']}")
        print(f"Bỏ qua / Abstain (0)     : {report['abstain']}")
        print(f"Điểm ròng (Correct-Wrong): {report['raw_points']}")
        print(f"Penalized Accuracy (PA)  : {report['penalized_accuracy'] * 100:.2f}%")
        print("-" * 60)
        print("Theo kiểu đáp án (kiểu gốc | tổng | đúng | sai | bỏ qua | PA):")
        for t_name, st in sorted(report["by_type"].items()):
            t_pa = (st["correct"] - st["wrong"]) / st["total"] * 100 if st["total"] else 0.0
            print(f"  {t_name:<13}| {st['total']:>5} | {st['correct']:>5} | {st['wrong']:>4} | {st['abstain']:>7} | {t_pa:6.1f}%")
        print("=" * 60)
        if args.eval_json:
            from src.report import write_json

            write_json(args.eval_json, report)
            print(f"✅ Đã xuất báo cáo chấm điểm: {args.eval_json}")

    if args.detail_json:
        from src.report import build_detail_records, write_json

        records = build_detail_records(details_results, questions, predictions=results, gold_map=gold_map or None)
        write_json(args.detail_json, records)
        print(f"✅ Đã xuất file chi tiết quyết định (json): {args.detail_json}")


if __name__ == "__main__":
    main()
