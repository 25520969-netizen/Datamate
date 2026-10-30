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

# 3. Chạy 300 câu benchmark và in trực tiếp log ra màn hình Colab
os.system(
    "python3 run_pipeline.py "
    "--questions train/sample_300_questions.jsonl "
    "--gold train/sample_300_answers.jsonl "
    "--output predictions_sample300.jsonl "
    "--backend openai "
    "--api-base http://localhost:8000/v1 "
    "--concurrency 8 "
    "--evaluate "
    "--fresh"
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
