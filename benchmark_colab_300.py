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

# 2.5 Tự động phát hiện thư mục dữ liệu trực tiếp trên Google Drive hoặc cục bộ
candidate_dirs = [
    Path("/content/drive/MyDrive/train_preprocessed_full"),
    Path("/content/drive/MyDrive/train_preprocessed"),
    Path("/content/drive/MyDrive/Datamate/train_preprocessed_full"),
    Path("/content/drive/MyDrive/Datamate/train_preprocessed"),
    Path("train_preprocessed_full"),
    Path("train_preprocessed"),
    Path("train/data"),
]
data_root = "train/data"
for p in candidate_dirs:
    if p.exists() and p.is_dir() and any(p.glob("biz_*")):
        data_root = str(p)
        print(f"🚀 Sử dụng dữ liệu trực tiếp từ: {data_root}\n")
        break
else:
    print(f"ℹ️ Sử dụng dữ liệu mặc định: {data_root}\n")

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
