# -*- coding: utf-8 -*-
"""
Khởi chạy vLLM trên Colab rồi chạy pipeline cho một tập câu hỏi bất kỳ.

Ví dụ:
  python colab_run.py                                  # benchmark 300 câu train (có chấm điểm)
  python colab_run.py --split public_test              # public_test, dữ liệu gốc public_test/data
  python colab_run.py --split public_test --data-root /content/drive/MyDrive/public_test_preprocessed
  python colab_run.py --questions X.jsonl --data-root Y --rulebook Z/rulebook.md

Rulebook và calendar được pipeline tự tìm cạnh dữ liệu / file câu hỏi (xem run_pipeline.resolve_split_files).
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train_sample", "train", "public_test"], default="train_sample")
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--rulebook", type=Path, default=None)
    ap.add_argument("--calendar", type=Path, default=None)
    ap.add_argument("--cache-root", type=Path, default=Path("/content/.cache/ingested") if Path("/content").exists() else None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/zaloAI/ketqua"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--shutdown", action="store_true", help="Ngắt runtime Colab khi chạy xong.")
    return ap.parse_args()


def split_defaults(split: str):
    if split == "train_sample":
        return ROOT / "train" / "sample_300_questions.jsonl", ROOT / "train" / "data", ROOT / "train" / "sample_300_answers.jsonl", "predictions_sample300.jsonl"
    if split == "train":
        return ROOT / "train" / "questions.jsonl", ROOT / "train" / "data", ROOT / "train" / "answers.jsonl", "predictions_train.jsonl"
    return ROOT / "public_test" / "questions.jsonl", ROOT / "public_test" / "data", None, "submission.jsonl"


def start_vllm() -> subprocess.Popen:
    for pkg in ("vllm", "duckdb"):
        if importlib.util.find_spec(pkg) is None:
            os.system(f"{sys.executable} -m pip install -q {pkg}")
    os.system("fuser -k 8000/tcp 2>/dev/null || true; pkill -9 -f vllm 2>/dev/null || true")
    print(f"⏳ Khởi động vLLM ({MODEL})...", flush=True)
    proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--port", "8000", "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90", "--enable-prefix-caching", "--enforce-eager", "--trust-remote-code",
    ])
    t0 = time.time()
    while os.system("curl -s http://localhost:8000/v1/models > /dev/null") != 0:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM thoát với mã {proc.returncode}")
        if time.time() - t0 > 900:
            raise TimeoutError("Quá 15 phút chờ vLLM khởi động")
        time.sleep(4)
    print(f"✅ vLLM sẵn sàng sau {time.time() - t0:.0f}s", flush=True)
    return proc


def main() -> None:
    args = parse_args()
    q, d, g, out = split_defaults(args.split)
    questions = args.questions or q
    data_root = args.data_root or d
    gold = args.gold or g
    output = args.output or Path(out)

    vllm = start_vllm()
    cmd = [
        sys.executable, str(ROOT / "run_pipeline.py"),
        "--questions", str(questions), "--data-root", str(data_root),
        "--output", str(output), "--details-output", str(output.with_name(output.stem + "_details.jsonl")),
        "--model", MODEL, "--backend", "openai", "--api-base", "http://localhost:8000/v1",
        "--concurrency", str(args.concurrency), "--request-timeout", "300", "--max-retries", "1",
    ]
    if gold is not None and Path(gold).exists():
        cmd += ["--gold", str(gold), "--evaluate"]
    for flag, val in (("--rulebook", args.rulebook), ("--calendar", args.calendar), ("--cache-root", args.cache_root)):
        if val is not None:
            cmd += [flag, str(val)]
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=False)
    finally:
        try:
            args.drive_dir.mkdir(parents=True, exist_ok=True)
            for f in (output, output.with_name(output.stem + "_details.jsonl")):
                if f.exists():
                    shutil.copy(str(f), str(args.drive_dir / f.name))
                    print(f"💾 Đã lưu {f.name} vào {args.drive_dir}")
        except Exception as exc:  # noqa: BLE001 - không có Drive thì bỏ qua
            print(f"ℹ️ Không lưu được lên Drive: {exc}")
        vllm.terminate()
        if args.shutdown:
            try:
                from google.colab import runtime  # type: ignore
                runtime.unassign()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
