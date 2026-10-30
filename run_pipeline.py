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
from model.baseline import DEFAULT_QWEN_7B_MODEL, DataMateAgent, QwenCoderClient


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
        "--backend",
        type=str,
        choices=["openai", "ollama", "transformers"],
        default="openai",
        help="Backend thực thi LLM Qwen 7B (mặc định: openai / vLLM).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_QWEN_7B_MODEL,
        help="Tên hoặc đường dẫn mô hình Qwen 7B (mặc định: Qwen/Qwen2.5-Coder-7B-Instruct).",
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
        default=2,
        help="Số lần tối đa tự sửa lỗi qua Sandbox Traceback (mặc định: 2).",
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
        default=6,
        help="Số lượng câu hỏi xử lý song song trên GPU continuous batching (mặc định: 6).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "train" / "answers.jsonl",
        help="Đường dẫn tới file answers.jsonl chuẩn (dùng khi có --evaluate).",
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

    questions = load_jsonl(args.questions)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    llm_client = QwenCoderClient(
        backend=args.backend,
        model_name=args.model,
        api_base=args.api_base,
    )

    agent = DataMateAgent(
        data_root=args.data_root,
        calendar_path=args.calendar,
        rulebook_path=args.rulebook,
        llm_client=llm_client,
        sandbox_timeout=args.timeout,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Nạp các câu trả lời tốt đã có từ trước (không phải abstain) nếu không bật cờ --fresh
    existing_cache: Dict[str, Dict[str, Any]] = {}
    if not args.fresh and args.output.exists():
        for ln in args.output.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                try:
                    item = json.loads(ln)
                    if item.get("type") != "abstain":
                        existing_cache[item["id"]] = item
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
    completed_count = 0

    for idx, q_obj in enumerate(questions):
        qid = q_obj["id"]
        if qid in existing_cache:
            results[idx] = existing_cache[qid]
            completed_count += 1

    to_run = [(idx, q_obj) for idx, q_obj in enumerate(questions) if results[idx] is None]
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

    tmp_output = args.output.with_suffix(".jsonl.tmp")
    lock = threading.Lock()
    start_time = time.time()

    def _flush_tmp() -> None:
        with tmp_output.open("w", encoding="utf-8") as out_f:
            for r in results:
                if r is not None:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _flush_tmp()

    def _solve_item(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        idx, q_obj = item
        pred = agent.answer_question(q_obj)
        return idx, pred

    if to_run:
        if args.concurrency > 1 and len(to_run) > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_solve_item, item): item for item in to_run}
                for fut in as_completed(futures):
                    idx, pred = fut.result()
                    with lock:
                        results[idx] = pred
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

                        log_line = (
                            f"[{completed_count}/{len(questions)}] ({completed_count/len(questions)*100:.1f}%) [MỚI] "
                            f"{qid} ({q_obj['business_id']}) -> {pred.get('type')}: {ans_preview} | Tốc độ: {avg_speed:.2f}s/câu | ETA: {rem_mins:.1f}m"
                        )
                        if done_new == 1 or done_new % 3 == 0 or completed_count == len(questions):
                            print(f"  {log_line}")
                            notify_colab(log_line)
        else:
            for idx, q_obj in to_run:
                pred = agent.answer_question(q_obj)
                results[idx] = pred
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

                log_line = (
                    f"[{completed_count}/{len(questions)}] ({completed_count/len(questions)*100:.1f}%) [MỚI] "
                    f"{qid} ({q_obj['business_id']}) -> {pred.get('type')}: {ans_preview} | Tốc độ: {avg_speed:.2f}s/câu | ETA: {rem_mins:.1f}m"
                )
                if done_new == 1 or done_new % 3 == 0 or completed_count == len(questions):
                    print(f"  {log_line}")
                    notify_colab(log_line)

    # Đổi tên file tạm thành file chính thức an toàn
    shutil.move(str(tmp_output), str(args.output))

    notify_colab("PIPELINE_COMPLETE_SHUTDOWN: Đã hoàn tất 100% toàn bộ câu hỏi! Đang tự động ngắt phiên Colab để tiết kiệm CU...")
    try:
        import subprocess
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
    except Exception:
        pass

    predictions = [r for r in results if r is not None]
    print(f"Đã xuất kết quả dự đoán tại: {args.output}")

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
