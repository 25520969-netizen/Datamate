#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script nhận file kết quả predictions_sample300.jsonl tự động từ Google Colab
qua webhook ntfy.sh sau khi Colab chạy xong 300 câu benchmark.
"""

import json
import time
from pathlib import Path
import requests

TOPIC = "zalo-datamate-results-kimvo"
OUTPUT_FILE = Path(__file__).resolve().parent / "predictions_sample300.jsonl"


def main() -> None:
    print(f"🎧 [Antigravity Listener] Đang chờ file kết quả từ Google Colab (topic: {TOPIC})...")
    start_time = int(time.time())

    for line in requests.get(f"https://ntfy.sh/{TOPIC}/json?since={start_time}", stream=True).iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if data.get("event") == "message" and "attachment" in data:
                    att = data["attachment"]
                    file_url = att["url"]
                    file_name = att["name"]
                    size = att.get("size", 0)
                    print(f"\n📥 Nhận được tín hiệu file: {file_name} ({size} bytes) từ Google Colab!")
                    print(f"⬇️ Đang tải về: {OUTPUT_FILE}...")
                    content = requests.get(file_url, timeout=30.0).content
                    OUTPUT_FILE.write_bytes(content)
                    print(f"✅ Đã lưu file kết quả thành công vào máy Mac: {OUTPUT_FILE} ({len(content)} bytes)!")
                    break
            except Exception as e:
                print("Lỗi nhận gói tin:", e)


if __name__ == "__main__":
    main()
