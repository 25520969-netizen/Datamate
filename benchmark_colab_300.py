import subprocess
import time
import os
import shutil
import importlib.util
from pathlib import Path

# 0. Tự động cài vLLM nếu môi trường chưa có
if importlib.util.find_spec("vllm") is None:
    print("📦 Đang tự động cài đặt vLLM trên Colab...")
    os.system("pip install -q vllm")

# 1. Khởi chạy vLLM ở nền (tối ưu prefix caching, GPU 90%)
os.system("fuser -k 8000/tcp 2>/dev/null || true; pkill -9 -f vllm 2>/dev/null || true")
print("⏳ [1/4] Đang khởi động vLLM Server trên GPU Colab...")
vllm = subprocess.Popen([
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
    "--port", "8000",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.90",
    "--enable-prefix-caching",
    "--enforce-eager"
])

# 2. Chờ vLLM sẵn sàng
while os.system("curl -s http://localhost:8000/v1/models > /dev/null") != 0:
    time.sleep(4)
print("✅ [2/4] vLLM đã sẵn sàng! Bắt đầu chạy benchmark 300 câu...\n")

# 2.5 Tự động phát hiện và giải nén train_preprocessed_full nếu có
data_root = "train/data"
drive_zip = Path("/content/drive/MyDrive/train_preprocessed_full.zip")
local_zip = Path("train_preprocessed_full.zip")
prep_dir = Path("train_preprocessed_full")

if not prep_dir.exists():
    if drive_zip.exists():
        print(f"📦 Phát hiện {drive_zip}! Đang tự động giải nén...")
        os.system(f"unzip -q -o '{drive_zip}' -d .")
    elif local_zip.exists():
        print(f"📦 Phát hiện {local_zip}! Đang tự động giải nén...")
        os.system(f"unzip -q -o '{local_zip}' -d .")

if prep_dir.exists():
    print("🚀 Sử dụng dữ liệu preprocessed đầy đủ: train_preprocessed_full\n")
    data_root = "train_preprocessed_full"
else:
    print("ℹ️ Sử dụng dữ liệu gốc: train/data\n")

# 3. Chạy 300 câu benchmark và in trực tiếp log ra màn hình Colab
os.system(
    f"python3 run_pipeline.py "
    f"--questions train/sample_300_questions.jsonl "
    f"--gold train/sample_300_answers.jsonl "
    f"--data-root {data_root} "
    f"--output predictions_sample300.jsonl "
    f"--backend openai "
    f"--api-base http://localhost:8000/v1 "
    f"--concurrency 8 "
    f"--evaluate "
    f"--fresh"
)

# 4. Lưu file kết quả về Google Drive
print("\n💾 [3/4] Đang lưu file kết quả về Google Drive...")
local_pred = Path("predictions_sample300.jsonl")
drive_pred = Path("/content/drive/MyDrive/predictions_sample300.jsonl")
if local_pred.exists():
    shutil.copy(str(local_pred), str(drive_pred))
    print(f"✅ Đã lưu kết quả tại: {drive_pred}")

# 5. Tắt GPU Colab
print("\n🎉 [4/4] Hoàn tất 100%! Đang tự động ngắt GPU Colab để tiết kiệm Compute Units...")
vllm.terminate()
time.sleep(2)
try:
    from google.colab import runtime
    runtime.unassign()
except Exception:
    pass
