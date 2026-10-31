import subprocess
import time
import os
import shutil
import importlib.util
from pathlib import Path

# 0. Tự động cài đặt thư viện cần thiết nếu môi trường chưa có
for pkg in ["vllm", "duckdb"]:
    if importlib.util.find_spec(pkg) is None:
        print(f"📦 Đang tự động cài đặt {pkg} trên Colab...")
        os.system(f"pip install -q {pkg}")

# 1. Khởi chạy vLLM ở nền (tối ưu prefix caching, GPU 90%)
os.system("fuser -k 8000/tcp 2>/dev/null || true; pkill -9 -f vllm 2>/dev/null || true")
print("⏳ [1/4] Đang khởi động vLLM Server trên GPU Colab (mô hình Qwen2.5-Coder-14B-Instruct-AWQ)...")
vllm = subprocess.Popen([
    "python3", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "--port", "8000",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.90",
    "--enable-prefix-caching",
    "--enforce-eager"
])

# 2. Chờ vLLM sẵn sàng (kèm kiểm tra tiến trình sống)
start_wait = time.time()
while os.system("curl -s http://localhost:8000/v1/models > /dev/null") != 0:
    if vllm.poll() is not None:
        raise RuntimeError(f"❌ vLLM server bị ngắt với exit code {vllm.returncode}! Vui lòng kiểm tra log lỗi GPU/VRAM.")
    if time.time() - start_wait > 600:
        raise TimeoutError("❌ Quá thời gian chờ vLLM khởi động (10 phút).")
    time.sleep(4)
print(f"✅ [2/4] vLLM đã sẵn sàng sau {time.time() - start_wait:.1f}s! Bắt đầu chạy benchmark 300 câu...\n")

# 2.5 Tự động phát hiện thư mục dữ liệu trực tiếp trên Google Drive hoặc cục bộ
candidate_dirs = [
    Path("/content/drive/MyDrive/train_preprocessed_full"),
    Path("/content/drive/MyDrive/Datamate/train_preprocessed_full"),
    Path("/content/drive/MyDrive/zaloAI/train_preprocessed_full"),
    Path("train_preprocessed_full"),
    Path("/content/drive/MyDrive/train/data"),
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
    f"--details-output predictions_sample300_details.jsonl "
    f"--model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ "
    f"--backend openai "
    f"--api-base http://localhost:8000/v1 "
    f"--concurrency 8 "
    f"--evaluate "
    f"--fresh"
)

# 4. Lưu 2 file kết quả (Chính thức + Chi tiết đối chiếu 2 Prompt) về Google Drive
print("\n💾 [3/4] Đang lưu 2 file kết quả về Google Drive...")
drive_dir = Path("/content/drive/MyDrive")
if drive_dir.exists():
    for fname in ["predictions_sample300.jsonl", "predictions_sample300_details.jsonl"]:
        local_f = Path(fname)
        drive_f = drive_dir / fname
        if local_f.exists():
            shutil.copy(str(local_f), str(drive_f))
            print(f"✅ Đã lưu {fname} tại: {drive_f}")
else:
    print("ℹ️ Thư mục /content/drive/MyDrive không tồn tại (chưa mount Drive). Các file kết quả được lưu tại thư mục hiện tại:")
    for fname in ["predictions_sample300.jsonl", "predictions_sample300_details.jsonl"]:
        if Path(fname).exists():
            print(f"   - {Path(fname).resolve()}")

# 5. Tắt GPU Colab
print("\n🎉 [4/4] Hoàn tất 100%! Đang tự động ngắt GPU Colab để tiết kiệm Compute Units...")
vllm.terminate()
time.sleep(2)
try:
    from google.colab import runtime
    runtime.unassign()
except Exception:
    pass
