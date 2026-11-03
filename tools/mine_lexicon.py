# -*- coding: utf-8 -*-
"""
Kiểm chứng và đo độ phủ của assets/lexicon.json trên bộ TRAIN (và chỉ train).

  python tools/mine_lexicon.py --train-root "/đường/dẫn/train" [--write]

Bảo vệ: chỉ chạy khi thư mục có `answers.jsonl` (đặc trưng của bộ train; public/private test không có đáp án)
và đường dẫn không chứa "public"/"private". Script KHÔNG đọc bất kỳ thư mục test nào.

Báo cáo:
  1. Bằng chứng cho từng viết tắt: số câu hỏi train chứa nó.
  2. Cách nói -> giá trị mã hoá: số shop train có giá trị biến thể trong cột phân loại.
  3. Kiểm chứng chéo THEO SHOP (chia đôi shop): trạng thái đơn và giá trị phân loại của nửa B có nhận dạng được
     chỉ bằng từ điển + luật hay không (độ phủ trên dữ liệu chưa thấy).
"""
import argparse
import collections
import json
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import normalize_order_status  # noqa: E402
from src.lexicon import DEFAULT_LEXICON_PATH, Lexicon, _strip  # noqa: E402

warnings.filterwarnings("ignore")
STATUS_KEYS = ("trạng thái", "tình trạng", "status", "tt đơn", "xử lý")
CANON = {"Hoàn thành", "Đã hủy", "Trả hàng", "Đang giao"}


def guard(train_root: Path) -> None:
    low = str(train_root).lower()
    if "public" in low or "private" in low or not (train_root / "answers.jsonl").exists():
        raise SystemExit("❌ Chỉ được dùng thư mục TRAIN (phải có answers.jsonl, đường dẫn không chứa public/private).")


def raw_frames(shop: Path):
    for f in sorted(shop.iterdir()):
        try:
            if f.suffix == ".csv":
                yield pd.read_csv(f, nrows=20000, low_memory=False)
            elif f.suffix == ".xlsx":
                yield from pd.read_excel(f, sheet_name=None, nrows=20000).values()
            elif f.suffix == ".sqlite":
                con = sqlite3.connect(f)
                for (t,) in con.execute("select name from sqlite_master where type='table'"):
                    yield pd.read_sql_query(f'select * from "{t}" limit 20000', con)
        except Exception:
            continue


def shop_values(shop: Path):
    status, cats = set(), set()
    for df in raw_frames(shop):
        for c in df.columns:
            if df[c].dtype != object:
                continue
            uniq = [u for u in df[c].dropna().astype(str).unique()[:300] if len(u) <= 40]
            if any(k in str(c).lower() for k in STATUS_KEYS):
                status.update(uniq)
            if 0 < len(uniq) <= 40:
                cats.update(_strip(u) for u in uniq)
    return status, cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON_PATH)
    ap.add_argument("--write", action="store_true", help="Ghi bằng chứng vào _meta.evidence của lexicon.json")
    args = ap.parse_args()
    guard(args.train_root)
    lex = Lexicon(json.loads(args.lexicon.read_text(encoding="utf-8")))
    questions = [json.loads(l)["question"] for l in open(args.train_root / "questions.jsonl", encoding="utf-8")]
    q_low = [q.lower() for q in questions]

    ev = {"abbreviations": {}, "value_concepts": {}}
    print("== 1. Bằng chứng viết tắt (số câu train chứa) ==")
    for tok in sorted(lex.always):
        n = sum(1 for q in q_low if re.search(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", q))
        ev["abbreviations"][tok] = n
    print({k: v for k, v in sorted(ev["abbreviations"].items(), key=lambda kv: -kv[1]) if v})
    print("  (0 câu train: giữ vì là viết tắt thông dụng chung, không có bằng chứng riêng):",
          [k for k, v in ev["abbreviations"].items() if v == 0])

    shops = sorted(p for p in (args.train_root / "data").iterdir() if p.is_dir())
    per_shop = {s.name: shop_values(s) for s in shops}

    print("\n== 2. Giá trị mã hoá trong dữ liệu ứng với cách nói (số shop train) ==")
    for name, spec in lex.value_concepts.items():
        wanted = {_strip(v) for v in spec["values"]}
        n_shop = sum(1 for _, (_, cats) in per_shop.items() if cats & wanted)
        n_q = sum(1 for q in q_low for ph in spec["phrases"] if re.search(r"(?<!\w)" + re.escape(ph) + r"(?!\w)", q))
        ev["value_concepts"][name] = {"shops": n_shop, "question_hits": n_q}
        print(f"  {name:18s} shop={n_shop:2d} câu_hỏi={n_q}")

    print("\n== 3. Kiểm chứng chéo theo shop (A = shop chẵn, B = shop lẻ) ==")
    A = [n for i, n in enumerate(per_shop) if i % 2 == 0]
    B = [n for i, n in enumerate(per_shop) if i % 2 == 1]
    seen_A = set().union(*[per_shop[n][0] for n in A])
    b_vals = set().union(*[per_shop[n][0] for n in B])
    b_only = b_vals - seen_A
    def recognized(v):  # chỉ từ điển + luật, không dùng dữ liệu của chính B
        return normalize_order_status(v) in CANON
    status_like = {v for v in b_vals if len(v) < 25 and not re.search(r"\d", v)}
    rec = {v for v in status_like if recognized(v)}
    print(f"  giá trị trạng thái ở B: {len(status_like)}, nhận dạng được: {len(rec)}")
    print(f"  chưa nhận dạng (xem có phải trạng thái đơn không): {sorted(status_like - rec)[:25]}")
    print(f"  giá trị chỉ có ở B (chưa thấy ở A) và nhận dạng được: {len({v for v in b_only if recognized(v)})}/{len({v for v in b_only if len(v)<25 and not re.search(chr(92)+'d', v)})}")
    cov = {}
    for name, spec in lex.value_concepts.items():
        wanted = {_strip(v) for v in spec["values"]}
        cov[name] = (sum(1 for n in A if per_shop[n][1] & wanted), sum(1 for n in B if per_shop[n][1] & wanted))
    print("  khái niệm giá trị (shop có ở A, shop có ở B):", cov)

    if args.write:
        data = json.loads(args.lexicon.read_text(encoding="utf-8"))
        data.setdefault("_meta", {})["evidence"] = ev
        args.lexicon.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nĐã ghi evidence vào", args.lexicon)


if __name__ == "__main__":
    main()
