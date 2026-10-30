#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Log Streamer & Auto-Shutdown Trigger for Google Colab
Đọc trực tiếp log tiến độ từ tiến trình pipeline trên Mac và đẩy lên ntfy.sh.
Khi pipeline chạy xong 100%, gửi tín hiệu PIPELINE_COMPLETE_SHUTDOWN để Colab tự tắt.
"""

import time
import requests
from pathlib import Path

LOG_FILE = Path("/Users/kimvo/.gemini/antigravity/brain/550707ae-569a-4fdd-b803-1c5c3afd9976/.system_generated/tasks/task-2488.log")
TOPIC = "zalo-datamate-kimvo-publictest"
NTFY_URL = f"https://ntfy.sh/{TOPIC}"


def send_ntfy(msg: str) -> None:
    try:
        requests.post(NTFY_URL, data=msg.encode("utf-8"), timeout=3.0)
    except Exception:
        pass


def main() -> None:
    print(f"📡 Đang theo dõi file log: {LOG_FILE}")
    print(f"🚀 Luồng log được đẩy lên: {NTFY_URL}")
    send_ntfy("🚀 BẮT ĐẦU THEO DÕI TIẾN TRÌNH PIPELINE QWEN-7B TRỰC TIẾP TỪ MÁY MAC...")

    last_pos = 0
    if LOG_FILE.exists():
        # Đọc 5 dòng gần nhất để đồng bộ ngay lập tức
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in lines[-5:]:
            if ln.strip():
                send_ntfy(ln.strip())
        last_pos = LOG_FILE.stat().st_size

    finished = False
    idle_cycles = 0

    while not finished:
        time.sleep(2.0)
        if not LOG_FILE.exists():
            continue

        curr_size = LOG_FILE.stat().st_size
        if curr_size > last_pos:
            idle_cycles = 0
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_data = f.read()
                last_pos = f.tell()

            for line in new_data.splitlines():
                ln = line.strip()
                if not ln:
                    continue
                print(f"[STREAM] {ln}")
                send_ntfy(ln)
                if "Penalized Accuracy (PA)" in ln or "PIPELINE_COMPLETE_SHUTDOWN" in ln:
                    finished = True
        else:
            idle_cycles += 1
            # Nếu 3 phút liên tục không có log mới và file predictions đã đủ 300 câu
            pred_file = Path("/Applications/zalo AI/predictions_sample300.jsonl")
            if pred_file.exists():
                lines_count = sum(1 for l in pred_file.read_text(encoding="utf-8").splitlines() if l.strip())
                if lines_count >= 300:
                    finished = True

    print("\n🎉 Đã hoàn tất toàn bộ pipeline!")
    send_ntfy("PIPELINE_COMPLETE_SHUTDOWN: Đã hoàn tất 100% toàn bộ câu hỏi! Đang tự động ngắt phiên Colab để tiết kiệm CU...")


if __name__ == "__main__":
    main()
