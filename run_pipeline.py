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
        default=PROJECT_ROOT / "train" / "calendar.csv",
        help="Đường dẫn tới file calendar.csv.",
    )
    parser.add_argument(
        "--rulebook",
        type=Path,
        default=PROJECT_ROOT / "train" / "rulebook.md",
        help="Đường dẫn tới file rulebook.md.",
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
        "--notify-topic",
        type=str,
        default="",
        help="Kênh ntfy.sh để stream log trực tiếp lên Google Colab và tự động ngắt session.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Số lượng câu hỏi xử lý song song trên GPU continuous batching (mặc định: 3).",
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
        choices=["adaptive", "strict"],
        help="Chính sách trọng tài: 'strict' (bắt buộc cả 2 thành công và đồng thuận, an toàn tối đa) hoặc 'adaptive' (fallback p1 khi p2 lỗi kỹ thuật).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Chạy mới toàn bộ câu hỏi, không tái sử dụng cache từ file output cũ.",
    )
    return parser.parse_args()


def main() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import shutil

    args = parse_args()

    # Tự động sửa lỗi typo đường dẫn nếu người dùng truyền nhầm .../data nhưng shop nằm ở thư mục cha
    if not args.data_root.exists() and args.data_root.name == "data" and args.data_root.parent.exists():
        if any(args.data_root.parent.glob("dev_*")) or any(args.data_root.parent.glob("biz_*")):
            print(f"ℹ️ [Path Correction] Phát hiện đường dẫn thừa '/data', tự động trỏ về: {args.data_root.parent}")
            args.data_root = args.data_root.parent

    # Tự động ưu tiên thư mục preprocessed nếu có trên Google Drive hoặc thư mục làm việc
    is_public_test = "public" in str(args.questions).lower() or "dev" in str(args.data_root).lower()
    if is_public_test:
        if str(args.data_root).rstrip("/\\").endswith("public_test/data") or not args.data_root.exists():
            candidate_public_preprocessed = [
                Path("/content/drive/MyDrive/public_test_preprocessed"),
                Path("/content/drive/MyDrive/Datamate/public_test_preprocessed"),
                Path("/content/drive/MyDrive/zaloAI/public_test_preprocessed"),
                PROJECT_ROOT / "public_test_preprocessed",
            ]
            for candidate in candidate_public_preprocessed:
                if candidate.exists() and candidate.is_dir() and any(candidate.glob("dev_*")):
                    print(f"🚀 [Auto-Detect] Tự động phát hiện dữ liệu public_test tiền xử lý tối ưu tại: {candidate}")
                    args.data_root = candidate
                    break
    elif str(args.data_root).rstrip("/\\").endswith("train/data") or not args.data_root.exists():
        candidate_preprocessed = [
            Path("/content/drive/MyDrive/train_preprocessed_full"),
            Path("/content/drive/MyDrive/train_preprocessed"),
            Path("/content/drive/MyDrive/Datamate/train_preprocessed_full"),
            Path("/content/drive/MyDrive/Datamate/train_preprocessed"),
            Path("/content/drive/MyDrive/zaloAI/train_preprocessed_full"),
            Path("/content/drive/MyDrive/zaloAI/train_preprocessed"),
            PROJECT_ROOT / "train_preprocessed_full",
            PROJECT_ROOT / "train_preprocessed",
        ]
        for candidate in candidate_preprocessed:
            if candidate.exists() and candidate.is_dir() and any(candidate.glob("biz_*")):
                print(f"🚀 [Auto-Detect] Tự động phát hiện dữ liệu tiền xử lý tối ưu tại: {candidate}")
                args.data_root = candidate
                break

    # Tự động định vị file questions nếu đường dẫn tương đối không nằm tại cwd
    if not args.questions.exists():
        if (PROJECT_ROOT / args.questions).exists():
            args.questions = PROJECT_ROOT / args.questions
        else:
            for p in [
                Path("/content/drive/MyDrive/zaloAI/data") / args.questions,
                Path("/content/drive/MyDrive/zaloAI") / args.questions,
                Path("/content/drive/MyDrive") / args.questions,
                Path("/content/drive/MyDrive") / args.questions.name,
            ]:
                if p.exists():
                    args.questions = p
                    break

    # Tương tự tự động định vị file gold
    if args.gold and not args.gold.exists():
        if (PROJECT_ROOT / args.gold).exists():
            args.gold = PROJECT_ROOT / args.gold
        else:
            for p in [
                Path("/content/drive/MyDrive/zaloAI/data") / args.gold,
                Path("/content/drive/MyDrive/zaloAI") / args.gold,
                Path("/content/drive/MyDrive") / args.gold,
                Path("/content/drive/MyDrive") / args.gold.name,
            ]:
                if p.exists():
                    args.gold = p
                    break

    # Tự động định vị calendar và rulebook
    if not args.calendar.exists() and (PROJECT_ROOT / args.calendar).exists():
        args.calendar = PROJECT_ROOT / args.calendar
    if not args.rulebook.exists() and (PROJECT_ROOT / args.rulebook).exists():
        args.rulebook = PROJECT_ROOT / args.rulebook

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
        )
    else:
        llm_client = QwenCoderClient(
            backend=args.backend,
            model_name=args.model,
            api_base=args.api_base,
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
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    details_path = args.details_output or args.output.with_name(args.output.stem + "_details.jsonl")
    tmp_output = args.output.with_suffix(".jsonl.tmp")
    tmp_details = details_path.with_suffix(".jsonl.tmp")

    # Nạp các câu trả lời tốt đã có từ trước (không phải abstain) nếu không bật cờ --fresh
    existing_cache: Dict[str, Dict[str, Any]] = {}
    existing_details: Dict[str, Dict[str, Any]] = {}
    if not args.fresh and args.output.exists():
        for ln in args.output.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    item = json.loads(ln)
                    if item.get("type") != "abstain":
                        existing_cache[item["id"]] = item
                except Exception:
                    pass
    if not args.fresh and details_path.exists():
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

    def notify_colab(msg: str) -> None:
        if args.notify_topic:
            try:
                import requests
                requests.post(
                    f"https://ntfy.sh/{args.notify_topic}",
                    data=msg.encode("utf-8"),
                    timeout=2.0,
                )
            except Exception:
                pass
        if args.api_base and "http" in args.api_base:
            try:
                import requests
                base_root = args.api_base.split("/v1")[0].rstrip("/")
                if "PIPELINE_COMPLETE_SHUTDOWN" in msg:
                    requests.post(f"{base_root}/shutdown", json={"msg": msg}, timeout=3.0)
                else:
                    requests.post(f"{base_root}/progress", json={"msg": msg}, timeout=1.0)
            except Exception:
                pass

    notify_colab(f"🚀 BẮT ĐẦU CHẠY PIPELINE (Đã bảo lưu {len(existing_cache)} câu tốt, tối ưu prompt L4, concurrency={args.concurrency})...")

    lock = threading.Lock()
    start_time = time.time()

    def _flush_tmp() -> None:
        with tmp_output.open("w", encoding="utf-8") as out_f:
            for r in results:
                if r is not None:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with tmp_details.open("w", encoding="utf-8") as out_d:
            for d in details_results:
                if d is not None:
                    out_d.write(json.dumps(d, ensure_ascii=False) + "\n")

    _flush_tmp()

    def _solve_item(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        idx, q_obj = item
        pred, details = agent.answer_question_dual(q_obj)
        return idx, pred, details

    if to_run:
        if args.concurrency > 1 and len(to_run) > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_solve_item, item): item for item in to_run}
                for fut in as_completed(futures):
                    idx, pred, details = fut.result()
                    with lock:
                        results[idx] = pred
                        details_results[idx] = details
                        completed_count += 1
                        _flush_tmp()

                        elapsed = time.time() - start_time
                        done_new = completed_count - (len(questions) - len(to_run))
                        avg_speed = elapsed / max(done_new, 1)
                        rem_secs = (len(to_run) - done_new) * avg_speed
                        rem_mins = rem_secs / 60.0

                        q_obj = questions[idx]
                        qid = q_obj["id"]
                        ans_preview = json.dumps(pred.get("answer"), ensure_ascii=False) if "answer" in pred else str(pred.get("type"))
                        if len(ans_preview) > 35:
                            ans_preview = ans_preview[:32] + "..."

                        if details.get("is_consensus"):
                            status_tag = "ĐỒNG THUẬN"
                        elif pred.get("type") != "abstain":
                            status_tag = f"FALLBACK ({details.get('decision_reason', 'adaptive')})"
                        else:
                            status_tag = "ABSTAIN"
                        log_line = (
                            f"[{completed_count}/{len(questions)}] ({completed_count/len(questions)*100:.1f}%) [{status_tag}] "
                            f"{qid} ({q_obj['business_id']}) -> {pred.get('type')}: {ans_preview} | Tốc độ: {avg_speed:.2f}s/câu | ETA: {rem_mins:.1f}m"
                        )
                        if done_new == 1 or done_new % 3 == 0 or completed_count == len(questions):
                            print(f"  {log_line}")
                            notify_colab(log_line)
        else:
            for idx, q_obj in to_run:
                pred, details = agent.answer_question_dual(q_obj)
                results[idx] = pred
                details_results[idx] = details
                completed_count += 1
                _flush_tmp()

                elapsed = time.time() - start_time
                done_new = completed_count - (len(questions) - len(to_run))
                avg_speed = elapsed / max(done_new, 1)
                rem_secs = (len(to_run) - done_new) * avg_speed
                rem_mins = rem_secs / 60.0

                qid = q_obj["id"]
                ans_preview = json.dumps(pred.get("answer"), ensure_ascii=False) if "answer" in pred else str(pred.get("type"))
                if len(ans_preview) > 35:
                    ans_preview = ans_preview[:32] + "..."

                status_tag = "ĐỒNG THUẬN" if details.get("is_consensus") else "ABSTAIN"
                log_line = (
                    f"[{completed_count}/{len(questions)}] ({completed_count/len(questions)*100:.1f}%) [{status_tag}] "
                    f"{qid} ({q_obj['business_id']}) -> {pred.get('type')}: {ans_preview} | Tốc độ: {avg_speed:.2f}s/câu | ETA: {rem_mins:.1f}m"
                )
                if done_new == 1 or done_new % 3 == 0 or completed_count == len(questions):
                    print(f"  {log_line}")
                    notify_colab(log_line)

    # Đổi tên file tạm thành file chính thức an toàn
    shutil.move(str(tmp_output), str(args.output))
    shutil.move(str(tmp_details), str(details_path))

    notify_colab("PIPELINE_COMPLETE_SHUTDOWN: Đã hoàn tất 100% toàn bộ câu hỏi! Đang tự động ngắt phiên Colab để tiết kiệm CU...")

    predictions = [r for r in results if r is not None]
    print(f"✅ Đã xuất kết quả dự đoán chính thức tại: {args.output}")
    print(f"✅ Đã xuất bảng chi tiết đối chiếu 2 prompt tại: {details_path}")

    if args.evaluate and args.gold.exists():
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
        print("=" * 60)

    if args.api_base and "http" in args.api_base:
        try:
            import requests
            base_root = args.api_base.split("/v1")[0].rstrip("/")
            if args.evaluate and "report" in locals():
                score_str = f"PA: {report['penalized_accuracy']*100:.2f}% | Đúng: {report['correct']} | Sai: {report['wrong']} | Bỏ qua: {report['abstain']}"
            else:
                score_str = f"Đã hoàn tất 100% ({len(questions)} câu)!"
            requests.post(f"{base_root}/shutdown", json={"msg": score_str}, timeout=5.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
