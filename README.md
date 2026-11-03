# DataMate 2.0: Hệ Thống Hỏi Đáp Dữ Liệu Tự Động Cho Doanh Nghiệp (Text-to-Code Pipeline)
**Zalo AI Challenge 2026 - DataMate Track**

> **Tóm tắt thông số kỹ thuật:**
> - **Mục tiêu tối thượng**: Tối đa hóa chỉ số Penalized Accuracy ($PA = \frac{C - W}{N}$) trên dữ liệu chưa từng thấy (Unseen Data) thông qua chiến lược Safe Abstain.
> - **Kiến trúc mô hình**: Hoạt động 100% Offline với mô hình ngôn ngữ mã nguồn mở tham số $\le 15\text{B}$ (khuyến nghị `Qwen2.5-Coder-14B-Instruct-AWQ` ~14.77B tham số).
> - **An toàn môi trường**: Môi trường Sandbox cách ly nghiêm ngặt, chặn đứng Arbitrary Code Execution (ACE) và các truy vấn mạng ra ngoài.
> - **Tài nguyên**: Tiêu thụ RAM thực tế kiểm soát dưới mức trần $8\text{ GB}$ (trung bình 50MB – 400MB RSS).
> - **Kiểm thử tự động**: 146 unit tests (chạy không cần GPU/LLM) cho các tầng tất định; chất lượng LLM chưa được đo trong thư mục này vì không kèm dữ liệu train.

---

## Khai báo mô hình sử dụng (theo luật cuộc thi)

| Mô hình | Tham số | Vai trò lúc inference | Ghi chú |
| :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | ~14.77 tỷ (lượng tử hoá AWQ 4-bit) | Sinh mã Python/DuckDB SQL (Prompt 1, 2, 3), trọng tài LLM (chế độ `tournament`), viết lại câu hỏi (tuỳ chọn `--rewrite-question`) | Mô hình duy nhất dùng lúc inference; trọng số công khai, phát hành trước 01/10/2026 |

- **Tổng tham số mọi mô hình lúc inference: ~14.77 tỷ ≤ 15 tỷ.** Không dùng embedding, reranker hay classifier nào; mọi bước còn lại (làm sạch dữ liệu, quy đổi đơn vị, giải ngày, so khớp đồng thuận, kiểm tra unanswerable) là mã tất định.
- Không fine-tune; không gọi API ngoài; LLM được phục vụ cục bộ (vLLM/Ollama) và không cần mạng.
- Nếu thay bằng mô hình khác (ví dụ bản không lượng tử hoá của cùng mô hình), cập nhật bảng này và giữ tổng ≤ 15 tỷ.

---

## 1. Giới thiệu tổng quan

**DataMate 2.0** là pipeline Text-to-Code và Text-to-SQL chuyển đổi câu hỏi bằng tiếng Việt tự nhiên sang mã thực thi truy vấn trực tiếp trên các nguồn dữ liệu thực tế của doanh nghiệp vừa và nhỏ (CSV, Excel, SQLite).

Hệ thống được tái cấu trúc triệt để theo các nguyên tắc kỹ thuật chuẩn mực:
1. **Chiến lược Safe Abstain (Penalized Accuracy Guard)**: Với thang điểm đúng $+1$, sai $-1$, từ bỏ $0$, việc đoán mò khi 2 mô hình bất đồng ý kiến mang kỳ vọng điểm âm. Hệ thống kích hoạt cơ chế Safe Abstain để bảo toàn điểm số một cách toán học.
2. **Sandbox an toàn & Phòng thủ Binder Error**: Môi trường thực thi in-process và subprocess được bảo vệ với danh sách trắng `__builtins__` và module an toàn; lớp bọc `SafeDuckDBWrapper` tự động ánh xạ các cột giả định (`status_norm`, `revenue_norm`, `qty_base`) về cột thực tế của bảng, triệt tiêu hoàn toàn lỗi DuckDB Binder Error.
3. **Chuẩn hoá tiếng Việt & Cell Grounding (DART-SQL)**: Bảo vệ nguyên vẹn các thực thể nhạy cảm (chuỗi trong ngoặc kép, kích cỡ F&B `(M)`, `(L)`, mã đơn `HD...`, `SP...`), mở rộng từ viết tắt kinh doanh phổ biến và trích xuất mẫu giá trị ô thực tế từ cơ sở dữ liệu để định hướng sinh mã.
4. **Làm sạch tự động chuẩn Mục 8 Rulebook**: Khử các dòng tổng cộng/tổng phụ (*subtotal rows*) mà không làm mất dòng hợp lệ, fill-down thuộc tính cấp đơn, khử dòng trùng lặp và unpivot bảng dạng ma trận.
5. **100% Khép kín & Ngoại tuyến**: Loại bỏ toàn bộ các yêu cầu HTTP/webhook ra bên ngoài, không phụ thuộc internet hay các dịch vụ cloud công cộng.

---

## 2. Cấu trúc thư mục dự án

```
Datamate/
├── README.md                 # Tài liệu kỹ thuật và hướng dẫn vận hành
├── requirements.txt          # Thư viện Python phụ thuộc
├── pytest.ini                # Cấu hình kiểm thử
├── .gitignore                # Bỏ qua dữ liệu lớn, cache, kết quả chạy
├── run_pipeline.py           # Điểm khởi chạy chính (đọc dữ liệu, chạy agent, ghi submission.jsonl)
├── evaluate.py               # Chấm điểm Penalized Accuracy
├── notebooks/
│   └── zaloAI_colab.ipynb    # Notebook Colab: bật vLLM, chạy 300/3000 câu train và 1000 câu public test
├── rule.md                   # Quy tắc làm việc cho trợ lý lập trình (không thuộc pipeline)
├── MULTI_STAGE_AGENT_DESIGN.md  # Tài liệu thiết kế kiến trúc ban đầu
├── assets/                   # Tài nguyên dùng chung
│   ├── calendar.csv          # Lịch âm - dương và ngày lễ Việt Nam
│   ├── rulebook.md           # Rulebook bộ train (chỉ làm dự phòng khi tập không kèm rulebook)
│   └── lexicon.json          # Từ điển ngôn ngữ - dữ liệu học từ train (xem mục 15)
├── model/
│   ├── __init__.py
│   └── baseline.py           # DataMateAgent, LLM client, prompt, đồng thuận, bỏ phiếu 2/3, Safe Abstain
├── src/                      # Các tầng xử lý
│   ├── __init__.py
│   ├── ingest.py             # Nạp dữ liệu gốc và preprocessed, làm sạch Mục 8, quy đổi đơn vị, schema prompt
│   ├── date_resolver.py      # Lịch âm-dương, mốc thời gian tương đối theo rulebook
│   ├── rule_engine.py        # Chọn mục rulebook, gợi ý kiểu đáp án, nhận diện câu unanswerable, few-shot
│   ├── lexicon.py            # Nạp và áp dụng lexicon.json
│   ├── telex.py              # Giải mã chữ gõ Telex theo luật
│   ├── vietnamese_normalizer.py  # Chuẩn hoá Unicode, viết tắt, viết lại câu hỏi (tuỳ chọn)
│   ├── postprocess.py        # Làm tròn Mục 9, chuẩn hoá kiểu, kiểm tra đồng thuận
│   └── sandbox.py            # Sandbox chạy mã sinh ra (in-process + subprocess)
├── tools/                    # Công cụ ngoài luồng chạy chính
│   ├── build_preprocessed_data.py  # Tạo thư mục <tập>_preprocessed từ dữ liệu gốc
│   └── mine_lexicon.py       # Kiểm chứng/đo độ phủ lexicon.json (chỉ đọc bộ train)
├── tests/                    # Unit test các tầng tất định (không cần GPU/LLM)
│   └── mock_data/            # Dữ liệu mẫu nhỏ cho test
├── train/                    # Bộ train: questions.jsonl, answers.jsonl, rulebook.md, calendar.csv (data/ bị gitignore)
└── public_test/              # Bộ public test: questions.jsonl, rulebook.md, calendar.csv (data/ bị gitignore)
```

---

## 3. Kiến trúc Pipeline 5 Tầng

```
[Câu hỏi tiếng Việt thô + as_of_date ("hôm nay")]
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: VIETNAMESE CONTEXT NORMALIZER & CELL GROUNDING                 │
│ - Chuẩn hoá Unicode NFC, sửa lỗi Telex dính phím                       │
│ - Bảo vệ thực thể: Giữ nguyên "chuỗi trong ngoặc kép", size (M)/(L)   │
│ - Mở rộng từ viết tắt kinh doanh: dt, sp, kh, đvt, hsd, sl, ck, v.v.   │
│ - DART-SQL Cell Grounding: Tìm mẫu giá trị ô thực tế từ bảng cho LLM   │
└────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: INGESTION, DATA CLEANING & SCHEMA ENRICHMENT                   │
│ - Làm sạch Mục 8 Rulebook: Nhận diện & loại bỏ dòng tổng phụ/subtotal  │
│ - Fill-down thuộc tính cấp đơn cho ô trống (Mã đơn, Ngày, Khách)       │
│ - Lọc bỏ dòng trùng lặp do lỗi xuất file Excel                         │
│ - Phát hiện & unpivot bảng ma trận (tháng/ngày thành hàng)             │
│ - Bổ sung cột định danh chuẩn: qty_base, status_norm, revenue_norm     │
└────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: DETERMINISTIC RULE INJECTION & FAST UNANSWERABLE               │
│ - DateResolver: Ánh xạ chuẩn mốc ngày (hôm nay, hôm qua, tuần này,     │
│   tháng trước, Tết âm lịch từ calendar.csv)                            │
│ - Deterministic Unanswerable Guard: Phát hiện câu hỏi ngoài kỳ ngày,   │
│   câu hỏi cảm tính/thái độ, kiến thức y tế ngoài bảng -> Trả về sớm    │
└────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: DUAL-PROMPT DIVERSE GENERATION & SECURE SANDBOX                │
│ - Prompt 1 (SQL Master): Sinh DuckDB SQL + CTEs chuẩn                  │
│ - Prompt 2 (Defensive CTE): Sinh truy vấn với CTE kiểm tra điều kiện   │
│ - Tự sửa lỗi 1 chu kỳ dựa trên lỗi traceback của Sandbox               │
│ - Thực thi trong Sandbox an toàn (Chặn builtins nguy hiểm, safe import)│
│ - SafeDuckDBWrapper: Phòng thủ Binder Error, fallback cột thông minh   │
└────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TẦNG 5: POST-PROCESSING & SAFE ABSTAIN CONSENSUS GATE                  │
│ - Làm tròn nửa đồng làm tròn lên (Mục 9 Rulebook)                      │
│ - Đồng thuận (P1 == P2) -> Chấp nhận đáp án                           │
│ - Trọng tài nghiệp vụ (Khách lẻ vs Khách thật) -> Ưu tiên P1           │
│ - Bất đồng ý kiến -> Trả về ABSTAIN (0 điểm, tránh bị trừ -1 điểm)     │
└────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
[Kết quả JSON chuẩn: {"id": "...", "type": "...", "answer": ...}]
```

---

## 4. Cơ chế An toàn Sandbox & Phòng thủ Lỗi Runtime

### 4.1. Khắc phục lỗ hổng thực thi mã tùy ý (Arbitrary Code Execution)
- Bộ Sandbox cũ sử dụng `__builtins__` mặc định và `ctypes.pythonapi.PyThreadState_SetAsyncExc` không an toàn, có nguy cơ gây Crash tiến trình hoặc rò rỉ quyền hệ thống.
- **Giải pháp mới**:
  - `SAFE_BUILTIN_NAMES`: Chỉ cho phép các built-in toán học và cấu trúc dữ liệu thuần túy (`len`, `range`, `int`, `float`, `str`, `list`, `dict`, `set`, `abs`, `sum`, `min`, `max`, v.v.). Chặn tuyệt đối `open`, `eval`, `exec`, `compile`, `breakpoint`, `help`, `input`.
  - `make_safe_import`: Whitelist import chỉ cho phép `math`, `re`, `json`, `datetime`, `date`, `timedelta`, `unicodedata`, `duckdb`, `pandas`, `numpy`. Chặn đứng `os`, `sys`, `subprocess`, `shutil`, `socket`, `requests`, `urllib`.
  - Bỏ hoàn toàn `ctypes` can thiệp luồng C-level; quản lý timeout thông qua `concurrent.futures` và cơ chế fallback tiến trình con có giới hạn thời gian.

### 4.2. Khắc phục DuckDB Binder Error trên các cột giả định
- Khi LLM sinh truy vấn `COALESCE("revenue_norm", "Thành tiền")` trên một bảng không có cột `"revenue_norm"`, DuckDB sẽ dừng lại với lỗi `Binder Error: Referenced column not found in FROM clause`.
- **Giải pháp mới**:
  - `SafeDuckDBWrapper` kiểm tra schema thực tế của bảng trong bộ nhớ. Nếu phát hiện các cột giả định (`status_norm`, `revenue_norm`, `qty_base`) không tồn tại, wrapper tự động viết lại biểu thức thành cột gốc tương đương (ví dụ `COALESCE(Thành tiền, Thành tiền)` hoặc `1=1` cho điều kiện trạng thái), giúp truy vấn chạy trơn tru mà không bị crash.

---

## 5. Chiến lược Safe Abstain & Công thức Penalized Accuracy

Theo luật cuộc thi, điểm số được tính bằng công thức:
$$\text{Penalized Accuracy (PA)} = \frac{C - W}{N}$$
Trong đó:
- $C$ là số câu trả lời **đúng** ($+1$ điểm).
- $W$ là số câu trả lời **sai** ($-1$ điểm).
- $N$ là tổng số câu hỏi trong tập đánh giá.
- Trả về **abstain** nhận $0$ điểm.

### Cơ sở toán học của Safe Abstain:
Giả sử khi hai prompt P1 và P2 bất đồng ý kiến, xác suất một trong hai đoán đúng là $p$. Kỳ vọng điểm số khi đoán mò là:
$$\mathbb{E}[\text{Guess}] = p \cdot (+1) + (1 - p) \cdot (-1) = 2p - 1$$
Khi hai mô hình khác nhau về kết quả, sự tự tin vào câu trả lời giảm xuống ($p < 0.5$). Do đó:
$$\mathbb{E}[\text{Guess}] < 0 \le \mathbb{E}[\text{Abstain}] = 0$$
Vì vậy, việc **từ bỏ (Abstain)** khi không có đồng thuận là chiến lược tối ưu toán học, bảo vệ điểm số của đội thi khỏi các cú phạt $-1$ oan uổng.

---

## 6. Hướng dẫn cài đặt & Kiểm thử

### 6.1. Khởi tạo môi trường ảo Python
Yêu cầu Python $3.9$ đến $3.11$:

```bash
# Đi vào thư mục pipeline
cd /Applications/pipeline_datamate

# Tạo môi trường ảo (ví dụ dùng python 3.9)
/usr/bin/python3 -m venv .venv
source .venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

### 6.2. Chạy bộ kiểm thử tự động (Unit Tests)
Bộ 146 tests kiểm chứng các module tất định (không cần LLM):

```bash
pytest tests/ -v
```

Kết quả mong đợi:
```
======================== 146 passed in 2.3s =========================
```

---

## 7. Hướng dẫn thực thi Pipeline

### 7.1. Chạy thử nghiệm nhanh không cần GPU (Mock Mode)
Kiểm tra luồng xử lý dữ liệu, DuckDB, Sandbox và định dạng JSON xuất ra mà không cần nạp mô hình LLM:

```bash
python run_pipeline.py \
    --questions train/questions.jsonl \
    --data-root train/data \
    --backend mock \
    --limit 5 \
    --arbitration strict \
    --output submission_mock.jsonl \
    --verbose
```

### 7.2. Chạy chính thức với local LLM (Offline Inference)
Khi chạy mô hình cục bộ qua vLLM hoặc Ollama (ví dụ `Qwen2.5-Coder-14B-Instruct-AWQ` trên cổng 8000):

```bash
python run_pipeline.py \
    --questions /path/to/questions.jsonl \
    --data-root /path/to/data \
    --backend openai \
    --api-base http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
    --arbitration strict \
    --concurrency 3 \
    --fresh \
    --output submission.jsonl
```

---

## 8. Bảng giải thích tham số dòng lệnh (CLI Reference)

| Tham số | Mặc định | Chức năng |
| :--- | :--- | :--- |
| `--questions` | `train/questions.jsonl` | Đường dẫn file danh sách câu hỏi (.jsonl). |
| `--data-root` | `train/data` | Thư mục chứa dữ liệu các doanh nghiệp (`biz_*`). |
| `--calendar` | `assets/calendar.csv` | File lịch tra cứu âm - dương. |
| `--rulebook` | `assets/rulebook.md` | File quy chuẩn nghiệp vụ. |
| `--output` | `submission.jsonl` | File kết quả đầu ra theo format nộp bài. |
| `--details-output` | `<output>_details.jsonl` | File lưu log chi tiết từng câu (mã sinh ra, kết quả P1/P2, lý do trọng tài). |
| `--backend` | `openai` | Backend LLM: `openai` (vLLM local), `ollama`, `transformers`, `mock`. |
| `--model` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | Tên mô hình cục bộ hoặc đường dẫn checkpoint. |
| `--api-base` | `None` | URL của endpoint local (ví dụ `http://localhost:8000/v1`). |
| `--arbitration` | `strict` | Chiến lược trọng tài: `strict` (Safe Abstain bảo toàn điểm số), `adaptive`, `tournament`. |
| `--no-tiebreak` | `False` | Tắt lá phiếu thứ ba (Prompt 3) khi P1/P2 bất đồng. Mặc định bật: chỉ nhận đáp án có ≥ 2/3 phiếu. |
| `--consensus-mode` | `dual` | `dual` (chạy 2 prompt so khớp đồng thuận) hoặc `single`. |
| `--max-retries` | `1` | Số lần tối đa cho phép sandbox gửi lỗi traceback để LLM tự sửa. |
| `--timeout` | `20` | Thời gian chạy tối đa của Sandbox cho mỗi đoạn mã (giây). |
| `--limit` | `0` | Giới hạn số câu hỏi cần chạy (0 = toàn bộ). |
| `--concurrency` | `4` | Số câu xử lý song song (mỗi câu 2 request P1/P2 → 8 sequence đồng thời). |
| `--rewrite-question` | `False` | Bật bước viết lại câu hỏi bằng LLM (thêm một lượt LLM tuần tự mỗi câu). |
| `--resume` | `False` | Tái dùng đáp án không-abstain trong file output cũ (chỉ khi tiếp tục lần chạy bị ngắt của CÙNG phiên bản code). Mặc định chạy mới toàn bộ. |
| `--fresh` | `False` | Giữ để tương thích; chạy mới đã là mặc định. |
| `--verbose` | `False` | Hiển thị log chi tiết mã nguồn và sandbox lên màn hình. |

---

## 9. Đánh giá chất lượng bằng Script Evaluate

Khi có file đáp án chuẩn (`answers.jsonl`), dùng `evaluate.py` để tính điểm:

```bash
python evaluate.py \
    --pred submission.jsonl \
    --gold /path/to/answers.jsonl \
    --output evaluation_report.json
```

File báo cáo `evaluation_report.json` sẽ thể hiện chi tiết:
- Số câu đúng ($C$)
- Số câu sai ($W$)
- Số câu từ bỏ (Abstain)
- Chỉ số Penalized Accuracy ($PA$)
- Bảng phân tích chi tiết theo từng loại câu hỏi (number, percent, date, boolean, string, list, table, unanswerable).

---

## 10. Giao thức Nâng cấp Chống Overfitting (PA $\ge$ 60% Protocol)

Hệ thống được nâng cấp toàn diện dựa trên **Tính bất biến Đại số & Nghiệp vụ (Algebraic & Domain Invariants)** nhằm đảm bảo tính tổng quát hóa tuyệt đối trên Unseen Data:

1. **Chuẩn hóa Ngày Tận gốc (`src/ingest.py`)**:
   - Tự động hoán vị định dạng ngày `MM/DD/YYYY` khi phát hiện vị trí tháng $> 12$ mà không hardcode tên cột.
   - Cưỡng chế tạo các cột chuẩn hóa `date_iso` (`YYYY-MM-DD`), `order_date_iso`, `timestamp_iso` trên toàn bộ các bảng, giúp DuckDB so sánh thời gian chuẩn xác theo thứ tự từ điển.
   - Hợp nhất các sheet Excel phân mảnh theo tháng/đợt bằng độ tương đồng Jaccard Schema ($\ge 0.70$), bảo tồn nguyên vẹn các bảng danh mục/bảng giá độc lập.

2. **Triệt tiêu Type-Mismatch & Khử Bẫy Số 0 (`src/postprocess.py`)**:
   - Tự động unpack các cấu trúc lồng nhau từ DuckDB `fetchall()` (`[(val,)]`, `[[val]]`) về dạng scalar đơn trị.
   - Bảo toàn tuyệt đối số 0 và 0.0%: Không chặn số 0 và không áp đặt các ngưỡng cứng phi thực tế (như > 20.000 số lượng hay > 500.000.000 doanh thu), đảm bảo hỗ trợ đầy đủ doanh nghiệp bán sỉ và chuỗi lớn.
   - Tích hợp **Post-Rounding Consensus**: Chấp nhận đồng thuận khi kết quả hai mô hình sau khi làm tròn theo quy tắc Mục 9 Rulebook (làm tròn nguyên cho tiền VND, 2 chữ số thập phân cho tỉ lệ/trung bình) là giống nhau.

3. **Bơm 4 Tiên đề Nghiệp vụ Bắt biến & Cell Samples (`model/baseline.py`)**:
   - Tiên đề 1: Chặn trên ngày hiện tại (`WHERE "date_iso" <= '{as_of_date}'`).
   - Tiên đề 2: Doanh thu thuần không trừ voucher cả đơn, không cộng VAT/phí ship.
   - Tiên đề 3: Khách lẻ không phải khách hàng (loại bỏ khi đếm/xếp hạng khách).
   - Tiên đề 4: Xếp hạng bán chạy luôn dựa trên sản lượng đơn vị cơ bản (`qty_base`), không theo doanh số tiền.
   - Cung cấp đúng 3 dòng dữ liệu mẫu (Cell Samples) cho mỗi bảng theo DART-SQL để LLM nhận diện chính xác định dạng ngày tháng và chuỗi ký tự thực tế.

4. **Vòng lặp Sửa lỗi Tự động Chuẩn DART-SQL (Execution-Guided Refinement) (`model/baseline.py`)**:
   - Phản hồi lỗi chi tiết từ DuckDB / Sandbox (syntax error, binder error) kèm gợi ý cột chính xác để LLM tự sửa.
   - Khi truy vấn chạy không lỗi nhưng trả về kết quả rỗng (None), hệ thống kích hoạt chẩn đoán điều kiện WHERE và cho phép mô hình thử lại thay vì bỏ cuộc ngay.

5. **Trọng tài Phân cấp Schema-Agnostic & Safe-Tournament Referee (`model/baseline.py`)**:
   - Loại bỏ hoàn toàn các chuỗi hardcode tên bảng (`"DanhSachHoaDonChiTiet"`).
   - Phát hiện hiện tượng nhân bản dòng do Cartesian product (`JOIN bloat`) khi một bên dùng JOIN có doanh thu $\ge 1.4$ lần bên không JOIN -> Ưu tiên bên không JOIN.
   - Trọng tài đối xứng hai chiều cho lọc trạng thái (`status_norm`) và đếm đơn (`COUNT(DISTINCT)`).
   - Prefix-Consensus cho câu hỏi Top K: Tự động cắt về đúng K phần tử đồng thuận.
   - **Safe-Tournament Referee (Agentar-Scale-SQL + Safe Abstain)**: Khi hai mô hình bất đồng ý kiến về số học, LLM Reasoning Referee sẽ thẩm định đối đầu dựa trên 4 Tiên đề Nghiệp vụ. NẾU KHÔNG CÓ VI PHẠM RÕ RÀNG HOẶC KHÔNG CHẮC CHẮN -> BẮT BUỘC TRẢ VỀ ABSTAIN (0 ĐIỂM) ĐỂ BẢO VỆ ĐIỂM SỐ, TUYỆT ĐỐI KHÔNG ĐOÁN MÒ BỪA BÃI!

6. **Phát hiện Miền Ngoài Phạm vi (Evidence-Based Unanswerable Guard) (`src/rule_engine.py`)**:
   - Nhận diện câu hỏi hỏi về các kỳ kinh doanh trong tương lai sau `as_of_date`.
   - Nhận diện các miền kiến thức nằm ngoài dữ liệu kinh doanh SME (nhiệt độ/độ ẩm kho, tác dụng phụ/dược lý y tế, khoảng cách địa lý GPS/Google Maps, thời tiết khí tượng, lịch vé tàu xe máy bay) dựa trên việc kiểm tra cấu trúc cột thực tế của shop.


---

## 11. Đối chiếu trung thực với các bài báo trong `papers/`

| Bài báo | Ý chính của bài báo | Pipeline thực sự làm gì | Đánh giá |
| :--- | :--- | :--- | :--- |
| **DART-SQL** (ACL Findings 2024) | (1) Viết lại câu hỏi dựa trên nội dung DB để khử mơ hồ; (2) Execution-Guided Refinement: đưa kết quả thực thi trở lại để sửa SQL lặp nhiều vòng. | `VietnameseNormalizer.canonicalize` viết lại câu hỏi bằng LLM kèm mẫu giá trị ô; vòng sửa lỗi chỉ kích hoạt khi lỗi/rỗng. | Áp dụng **một phần**. Từ bản này câu viết lại chỉ là gợi ý cho LLM; các bước tất định (ngày, kiểu đáp án, unanswerable) dùng câu gốc. |
| **Agentar-Scale-SQL** | Sinh nhiều ứng viên đa dạng, **gom nhóm theo kết quả thực thi**, rồi đấu loại vòng tròn từng cặp bằng selector huấn luyện RL. | Trước đây: trọng tài LLM chỉ chạy ở chế độ `adaptive`/`tournament` (mặc định `strict` không bao giờ gọi), không có selector huấn luyện. Bản này: thêm lá phiếu thứ ba + gom nhóm theo kết quả thực thi, nhận đáp án ≥ 2/3. | Trước: **gắn nhãn sai**. Nay: áp dụng phần gom nhóm theo kết quả thực thi; không có selector RL. |
| **DAIL-SQL** | DAIL-O: ví dụ dạng cặp câu hỏi–SQL không kèm schema; DAIL-S: chọn ví dụ theo độ tương đồng câu hỏi đã che thực thể + khung SQL dự đoán. | 17 ví dụ viết tay, chọn bằng luật từ khoá. | DAIL-O: **có**. DAIL-S: **không** (chỉ là luật từ khoá). Các ví dụ cũ còn sai luật nghiệp vụ (lọc `= 'Hoàn thành'` làm mất đơn đang giao, công nợ không theo từng đơn) — đã sửa. |
| **ViLexNorm** (EACL 2024) | Một **kho ngữ liệu** chuẩn hoá từ vựng tiếng Việt mạng xã hội và các mô hình baseline (BARTpho...). | Danh sách luật Telex/viết tắt viết tay; không dùng ngữ liệu hay mô hình nào của bài báo. | **Không áp dụng**; chỉ lấy cảm hứng. Một số luật cũ còn làm sai nghĩa ("nhiu"→"bao nhiêu", "ck"→"chiết khấu", "tk"→"tồn kho") — đã sửa. |
| **DIN-SQL** | Phân rã: schema linking → phân loại độ khó → sinh SQL → tự sửa. | Không có module tương ứng trước đây. Prompt 3 (plan-then-code) nay mượn ý phân rã kế hoạch trước khi viết SQL. | Trước: **chỉ trích dẫn**, chưa áp dụng. |
| **XiYan-SQL** | M-Schema + nhiều generator + mô hình chọn ứng viên. | Schema dạng `CREATE TABLE` (không phải M-Schema), không có mô hình chọn. | **Chỉ trích dẫn**, chưa áp dụng. |

Code còn nhắc tới CHESS, BIRD, "Borchmann 2025" — các tài liệu này không có trong `papers/`.

## 12. Nhật ký sửa lỗi (bản rà soát tổng quát hoá)

Các lỗi dưới đây đều được tái hiện bằng script trước khi sửa; không có quy tắc nào gắn với một shop cụ thể.

1. **Luật đơn hợp lệ sai trong mọi prompt và ví dụ**: lọc `"status_norm" = 'Hoàn thành'` bỏ mất đơn *đang giao* (Rulebook §1: đơn đang giao vẫn tính). Nay dùng `NOT IN ('Đã hủy', 'Trả hàng')`. Trên dữ liệu thử: 390.000 (sai) → 480.000 (đúng).
2. **Đồng thuận giả**: `check_consensus` coi 12.4% và 12.0% (hay trung bình 2.45 và 2.1) là khớp vì so sau khi làm tròn về số nguyên; nhánh `post_rounding_consensus` còn làm tròn trung bình về số nguyên. Nay so ở 2 chữ số thập phân.
3. **List số bị khử trùng**: `[3, 3]` (3 thùng 3 chai) thành `[3]`. Nay chỉ khử trùng tên.
4. **Phần trăm < 1% bị nhân 100**: 0.45% thành 45%. Nay chỉ quy đổi khi code không nhân 100.
5. **"Trà sữa (M)" và "Trà sữa (L)" bị coi là đồng thuận**. Nay hậu tố size phải khớp.
6. **Trọng tài "khách lẻ" so khớp chuỗi con**: dấu hiệu `"le"` khớp với mọi đáp án chứa "le" (vd "Lemon tea") và bỏ qua kiểm tra đồng thuận. Nay so khớp chính xác nhãn.
7. **List khác độ dài vẫn được nhận** (cắt về độ dài ngắn hơn khi không có top-k). Nay: top-k (kể cả "3 món ... nhất") được cắt về k trước khi so; ngoài ra phải bằng nhau.
8. **Tết không ghi năm** trả về cả Tết sắp tới → kiểm tra tất định gắn nhầm *unanswerable*; kỳ "trước Tết" luôn được thêm vào. **"Cùng kỳ năm ngoái"** bị hiểu thành cả năm ngoái. Đã sửa.
9. **Khoảng ngày dữ liệu**: nay đọc khoảng ngày README công bố (Rulebook §7) khi có.
10. **Chuẩn hoá câu hỏi**: bỏ map sai nghĩa; sửa lỗi cache trộn thực thể giữa hai câu chỉ khác "(M)"/"(L)"; câu viết lại phải giữ đủ thực thể đã che.
11. **Sandbox**: code sinh ra có thể sửa tại chỗ DataFrame cache dùng chung cho mọi câu hỏi/luồng → nay nhận bản sao lười. `date_context["last_month"]` từng âm thầm trả về *tháng này* → nay ánh xạ đúng hoặc báo `KeyError` để vòng sửa lỗi chạy.
12. **Ingest**: cột cấp đơn như "Mã KH" không được fill-down cho dòng sau của đơn (khách bị coi là khách lẻ khi xếp hạng theo mã) → nay nhận diện theo dữ liệu các cột chữ bất biến trong mỗi đơn. Dòng chi phí "Tổng vệ sinh máy lạnh" bị xoá nhầm như dòng tổng → nay chỉ xoá khi dòng chỉ có nhãn tổng. Tăng `CACHE_VERSION` (hiện `v16`) để không dùng lại cache cũ.
13. **Schema prompt**: cột ngày được khai báo `TIMESTAMP` trong khi là chuỗi; mẫu dòng chứa toàn alias tổng hợp bị cắt cụt. Nay khai báo đúng kiểu, liệt kê giá trị cột phân loại (≤ 8 giá trị), ẩn alias trùng.
14. **Bỏ phiếu 2/3**: khi P1/P2 bất đồng hoặc một bên lỗi, Prompt 3 (plan-then-code) làm lá phiếu thứ ba; chỉ nhận đáp án có ≥ 2 phiếu, còn lại vẫn Safe Abstain.
15. **Đáp án chuỗi được đối chiếu với giá trị ô thật**: nếu bản đã cắt chú thích không có trong dữ liệu mà bản gốc có (vd "Bia Tiger (24 lon)") thì giữ bản gốc.
16. **`evaluate.py`** nay chấm đúng luật chính thức (không bỏ khoảng trắng khi so chuỗi; phần trăm so sau khi làm tròn 2 chữ số) để PA nội bộ không bị thổi phồng.
17. **`run_pipeline.py`**: trước đây mặc định tái dùng đáp án của lần chạy cũ (có thể từ phiên bản code cũ); nay chỉ khi `--resume`.

### Đợt rà soát 2

18. **Quy đổi đơn vị cơ bản chưa từng chạy lúc suy luận**: `qty_base` chỉ là bản sao "Số lượng"; quy đổi chỉ có trong script offline `tools/build_preprocessed_data.py`. Nay `ingest` quy đổi khi nạp dữ liệu, đọc từ bảng quy đổi (theo mã hoặc tên hàng) hoặc dòng "1 thùng = 4 lốc, 1 lốc = 6 lon" trong README, hỗ trợ nhiều cấp, và thêm cột `Đơn vị cơ bản` (Rulebook §3, §10).
19. **Trạng thái thanh toán bị lấy làm trạng thái đơn**: nếu cột "Trạng thái thanh toán" đứng trước "Tình trạng đơn", mọi đơn "Đã trả"/"Chưa trả" thành "Trả hàng" (chuỗi chứa "trả") → doanh thu bằng 0. "Đang kiểm tra" cũng thành "Trả hàng" (chứa "tra"). Nay một hàm chuẩn hoá duy nhất chỉ nhận cụm "trả hàng/trả lại/hoàn hàng..." và chọn cột trạng thái đơn theo tỉ lệ giá trị nhận dạng được, bỏ cột thanh toán.
20. **Ngày MM/DD**: trước chỉ đảo ở ô có phần thứ hai > 12, các ô còn lại cùng cột ("03/04/2026") bị đọc ngược. Nay nhận định theo cả cột.
21. **Bộ chọn luật cắt ở 2500 ký tự**: câu lãi gộp mất Mục 5, câu nhà thuốc mất Mục 1; Mục 7, 9, 10 chưa từng vào prompt. Nay ưu tiên luật ngành > mục chủ đề > trích Mục 9 > Mục 1 > Mục 3 > Mục 7/10/2/8 trong trần 7500 ký tự.
22. **Unanswerable nhầm**: "Khách Nam mua..." bị coi là câu hỏi giới tính; câu đánh giá sao/thời tiết/khoảng cách bị từ chối kể cả khi shop có cột tương ứng. Nay kiểm tra cột trước khi kết luận. Câu tồn kho nhận cả cột "Tồn đầu kỳ"/bảng kho.
23. **Kiểu đáp án**: "Trong các món, món nào ... nhất?" bị gợi ý `list` (đúng là `string`); tên có dấu phẩy ("Bánh mì thịt, chả") bị cắt thành list chỉ vì câu có chữ "các"; câu "tháng nào" trả `YYYY-MM-DD` thay vì `YYYY-MM`. Đã sửa.
24. **Trạng thái "Trả hàng" lan sang bảng khác chỉ vì trùng số mã** (phiếu nhập "1" và đơn bán "1"). Nay chỉ lan sang bảng cùng không gian mã đơn.
25. **`revenue_norm`** không còn ưu tiên "Tổng thanh toán" (gồm VAT/phí giao, đã trừ giảm giá cả đơn) làm doanh thu.

### Đợt rà soát 3

26. **Gộp tên bằng cách bỏ dấu** làm nhập hai mặt hàng/khách khác nhau: "Bánh mì bơ" → "Bánh mì bò", "Anh Bào" → "Anh Bảo". Nay chỉ gộp các cách viết khác hoa/thường, khoảng trắng, dấu câu; bản gõ không dấu chỉ gộp khi có duy nhất một bản có dấu tương ứng.
27. **Số kiểu Việt Nam**: `pandas.read_csv` đọc "55.000" (55 nghìn đồng) thành 55.0; parser cũng đọc chuỗi "1.500" thành 1.5; "1,5" (thập phân dấu phẩy) không đọc được nên cột thành chuỗi. Nay nhận dấu chấm phân cách nghìn theo cả cột (cột tiền hoặc có giá trị nhiều nhóm) và nhận dấu phẩy thập phân.
28. **Bộ giải ngày**: "tháng 3 năm ngoái" ra tháng 3 năm nay cộng cả năm ngoái; "từ 1/5/2026 đến 10/5/2026" thêm kỳ thừa "cả tháng 5"; "tuần trước nữa" thành tuần trước. Bổ sung "quý này/quý trước", "từ đầu năm", "năm YYYY", "ngày D/M" và "từ D/M đến D/M" không ghi năm.
29. **Làm tròn tiền**: câu hỏi tiền không có từ khoá cũ ("khách chi bao nhiêu") được làm tròn 2 chữ số thay vì về đồng.

### Đợt rà soát 4

30. **Cache parquet dùng chung giữa các tập**: thư mục cache mặc định là `<data_root>/../../.cache` và chỉ khoá theo `business_id`, nên tập test có `biz_001` nhận nhầm dữ liệu train `biz_001` đã cache; dữ liệu nguồn thay đổi cũng không làm mất hiệu lực cache. Nay manifest lưu dấu vân tay (đường dẫn tuyệt đối + tên/kích thước/mtime file).
31. **Excel nhiều sheet**: sheet đầu là "Danh mục" thì các sheet tháng cùng cấu trúc không được gộp (LLM phải tự UNION). Nay gom cụm theo cấu trúc, không phụ thuộc thứ tự sheet.
32. **`Decimal` của DuckDB** (vd `100.0 * x`, cột DECIMAL) bị serialize thành chuỗi `"3.75"`, nên list/bảng số thành list/bảng chuỗi. Nay ra số JSON ở cả sandbox trong tiến trình lẫn tiến trình con.
33. **Sai kiểu do code gán nhầm `type`**: câu "tỉ lệ" gán `number`, "ngày nào" trả chuỗi ngày, "có ... không" trả "có", câu một đối tượng trả list một phần tử. Nay hiệu chỉnh khi giá trị chuyển được không mơ hồ sang kiểu câu hỏi đòi.
34. **Một exception bất ngờ ở một câu làm sập cả lượt chạy** và file submission chính không được ghi. Nay câu đó abstain, lượt chạy tiếp tục.

### Gỡ gắn cứng (chống overfit)

Nguyên tắc: luật nghiệp vụ chỉ lấy từ rulebook đang dùng (mỗi tập dữ liệu có thể có rulebook khác); code không chứa id shop, tên bảng hay câu hỏi cụ thể của bộ train.

35. **Prompt**: bỏ khối luật chép từ rulebook train (đơn đang giao, trả một phần, COD, phí phục vụ, khách lẻ, cách tính công nợ, tỉ lệ huỷ...). Prompt chỉ giữ chuẩn định dạng đáp án và thứ tự xếp hạng của **đề thi**, cùng hướng dẫn kỹ thuật SQL; luật lấy từ đoạn rulebook chèn vào. Trọng tài LLM cũng đọc luật từ rulebook thay vì luật viết sẵn.
36. **Cấm chú thích trong code sinh ra**: cả ba prompt cấm `#`, `--`, `/* */`, docstring; Prompt 3 không còn yêu cầu docstring kế hoạch mà phân rã bằng CTE đặt tên (`ky_hoi`, `ban_ghi_hop_le`, `tong_hop`). Ví dụ few-shot viết lại không còn chú thích SQL.
37. **Few-shot**: các ví dụ cũ nhúng luật train (lọc trạng thái, giờ ca 06–13h/14–21h, công nợ theo đơn). Nay chỉ minh hoạ cú pháp DuckDB, điều kiện nghiệp vụ để chỗ trống `<... theo rulebook>`.
38. **Bộ chọn luật**: nhận mục rulebook theo từ khoá tiêu đề thay vì số mục cố định; bỏ đoạn "footer" chép cứng luật thời gian; luôn đưa phần mở đầu và mục Thời gian của rulebook thật.
39. **Bộ giải ngày** đọc ngày đầu tuần và khoảng Tết từ rulebook (`RulebookKnobReader` trước đây đọc nhưng không nơi nào dùng). Mốc thời gian chuyển từ "BẮT BUỘC" sang "tham khảo, rulebook là chuẩn".
40. **Luật phụ thuộc rulebook chỉ bật khi rulebook ghi**: loại "Khách lẻ" khỏi đáp án/xếp hạng; "lãi gộp thiếu cột giá vốn → không trả lời được" (chỉ khi rulebook cấm tự tính lại giá vốn).
41. **Heuristic tinh chỉnh theo train trong trọng tài** đã bỏ: ưu tiên tỉ lệ huỷ "0 < x ≤ 50%", ưu tiên bên không JOIN khi số tiền "≥ 1.4 lần".
42. **Từ điển lỗi gõ chép từ câu hỏi train** (`naof`, `cuwar`, `bani`, `quarr`...) và các cụm đặc thù trong bộ suy luận kiểu (`từng người nha`, `thang naof`, `ees`, `tuổi nợ`) đã bỏ; lỗi gõ để bước viết lại câu hỏi bằng LLM xử lý chung.
43. **So khớp tên khi đồng thuận** dùng đúng chuẩn hoá của luật chấm (giữ dấu), bỏ từ điển "từ ngành hàng bỏ qua" (`bia`, `bột giặt`...) lấy từ tên hàng train.
44. **Ingest**: bỏ danh sách tên bảng danh mục cứng (`DanhSachSanPham`, `BanHang_HangHoa`...) → nhận bảng danh mục theo cấu trúc; bỏ bảng `tra_hang`/cột `MaDon` → nhận bảng phiếu trả theo từ vựng trong tên bảng; bỏ regex gộp bảng tháng chỉ cho tên `BanHang_*` → gộp mọi bảng cùng cấu trúc chia theo kỳ/đợt; bỏ id shop (`biz_1xx`, `dev_0xx`) và tên file train trong comment/docstring.
45. **`run_pipeline.py` không còn tự đổi `--data-root`** sang thư mục `*_preprocessed*` hay dò thư mục theo tiền tố id shop `dev_*`/`biz_*`. `tools/build_preprocessed_data.py` dùng chung hàm chuẩn hoá trạng thái của pipeline; `notebooks/zaloAI_colab.ipynb` (trước là `benchmark_colab_300.py`) chạy được cho mọi tập.

Còn giữ có chủ đích: giá trị mặc định của `RulebookKnobs` (thứ Hai, 23 tháng Chạp – mùng 10 tháng Giêng, ca 06:00–13:59) chỉ dùng khi rulebook không ghi; các danh sách từ vựng tiếng Việt chung (tên cột tiền/đơn vị/trạng thái, viết tắt nghiệp vụ); luật unanswerable theo từ khoá nhưng chỉ kích hoạt khi shop không có cột liên quan.

**Chưa đo được**: thư mục không kèm dữ liệu train hay LLM, nên mức tăng PA thực tế cần đo lại bằng notebook `notebooks/zaloAI_colab.ipynb` trên GPU.

---

## 13. Tối ưu thời gian chạy mỗi câu (vLLM, GPU ~22GB)

Code không chứa cấu hình GPU; các tối ưu nằm ở phía pipeline:

1. **Bố cục prompt cho prefix caching**: system prompt = nguyên tắc chung + **toàn bộ rulebook**, giống hệt cho mọi câu hỏi/mọi vai trò (không còn `as_of_date` hay câu hỏi trong system) → chỉ prefill một lần cho cả tập. User prompt theo thứ tự schema shop → câu hỏi → khối vai trò P1/P2/P3 ở **cuối**, nên P1/P2/P3 cùng câu và các câu cùng shop dùng chung phần đầu. README trong schema không còn lọc theo câu hỏi để phần đầu schema ổn định. Đo trên dữ liệu thử: system ~16.9k ký tự dùng chung, user ~3.8k ký tự trong đó ~3.4k chung giữa P1 và P2.
2. **Dừng sinh ngay khi đóng khối code** (`stop = ["\n```\n"]`): bỏ phần lời giải thích mô hình hay viết thêm sau code. Mẫu stop không khớp dấu mở "```python".
3. **Không chú thích trong code** (prompt + few-shot): ít token sinh hơn.
4. **Bỏ lượt LLM tuần tự viết lại câu hỏi** (mặc định tắt, bật bằng `--rewrite-question`); kiểm tra unanswerable tất định chạy **trước** mọi lời gọi LLM.
5. **Mỗi câu chỉ còn 1 vòng LLM song song (P1 ‖ P2)** khi đồng thuận; Prompt 3 chỉ chạy khi bất đồng.
6. `--concurrency` mặc định 4 (8 sequence đồng thời).

**Yêu cầu khi khởi chạy vLLM**: system prompt dài (~6–7k token với rulebook hiện tại) nên `--max-model-len` cần ≥ 12288 (khuyến nghị 16384) và bật `--enable-prefix-caching`; `--max-model-len 8192` sẽ bị lỗi độ dài.

---

## 14. Rulebook theo từng tập & giải mã Telex (rà soát trên dữ liệu thật)

Cấu trúc mỗi tập: `<split>/{data/, questions.jsonl, rulebook.md, calendar.csv}`. **Rulebook của public test khác train** (thêm luật như "giờ cao điểm", "khách trung thành", "topping ghi chung ô", mục "Mẹ và bé"...), nên chạy sai rulebook sẽ sai luật.

46. **`run_pipeline.py` tự tìm `rulebook.md` và `calendar.csv`** theo thứ tự: tham số `--rulebook`/`--calendar` → `--data-root` → thư mục cha của data-root → thư mục chứa file câu hỏi. Đường dẫn chỉ định mà không tồn tại, hoặc không tìm thấy rulebook, đều **dừng báo lỗi** (không còn âm thầm dùng bản train; chỉ khi thử nghiệm mới dùng `--allow-default-rulebook`). Lúc chạy in ra đường dẫn và dòng tiêu đề rulebook đang dùng.
47. **Giải mã Telex theo luật** (`src/telex.py`): ~8% câu hỏi train và ~2% public có chữ gõ Telex thô ("thangs", "tieenf", "dduwowcj"). Bộ giải mã áp luật gõ Telex (không có danh sách từ lỗi) và chỉ đổi từ khi kết quả là âm tiết tiếng Việt hợp lệ; không đổi chuỗi trong ngoặc kép, từ viết hoa giữa câu, từ thuộc dữ liệu của shop (tên hàng, tên cột). Kiểm tra trên toàn bộ câu hỏi thật: các từ bị đổi đều là lỗi Telex đúng nghĩa; lần đầu có đổi nhầm "tra/tro/tri" → "tả/tỏ/tỉ" và "triệu" → "tỉệu" nên đã siết điều kiện: phím dấu phải đứng sau nguyên âm.
48. **Đo trên dữ liệu train thật (3000 câu, không cần LLM)**:
    - Gợi ý kiểu đáp án (`infer_expected_type`) đúng **97.1%** (loại câu unanswerable, vốn không suy ra được từ câu chữ: 2.3% bộ train).
    - Phát hiện *unanswerable* tất định: bản đầu **32 đúng / 4 báo nhầm** trên 70 câu unanswerable. Sau sửa: **47 đúng / 0 báo nhầm**. Các lỗi sửa: "nhóm *dị ứng*" là tên nhóm hàng chứ không phải câu hỏi y khoa (kiểm tra giá trị thật trong dữ liệu shop); câu hết hạn có ngày tương lai bị coi là ngoài khoảng dữ liệu; cột "Tồn nhỏ nhất/lớn nhất/định mức" là ngưỡng, không phải số tồn; từ khoá so cả dạng không dấu; thêm câu so sánh với tháng liền trước / cùng kỳ năm trước mà kỳ so sánh nằm trước dữ liệu của shop; "khách nam" viết thường là giới tính, "khách Nam" viết hoa là tên người. Trên public test: 8/1000 câu bị gắn unanswerable, đọc qua đều là câu hỏi giới tính/địa chỉ/lý do huỷ/thái độ.
    - Còn 23/70 câu unanswerable chưa bắt được bằng luật; các câu này phụ thuộc cấu trúc dữ liệu từng shop (không có kênh bán, không có mã khách...) nên để LLM (đồng thuận ≥ 2/3 trả `unanswerable`) xử lý.
49. `get_shop_date_range` được tính một lần mỗi shop (trước đây quét lại toàn bộ bảng mỗi câu, ~0.5 giây/câu trên shop lớn).
50. **Ngân hàng chủ đề "dữ liệu không có thông tin này"** (`RuleEngine.UNANSWERABLE_TOPICS`, khai báo dạng bảng, mở rộng bằng từ vựng tiếng Việt chung, không lấy từ câu hỏi của một tập): đánh giá/thái độ, y khoa, lý do huỷ/trả, tuổi/ngày sinh, nghề nghiệp/gia đình, liên hệ mạng xã hội, quê quán/quốc tịch, bảo quản, khoảng cách/bản đồ, thời tiết, lịch vận tải, đối thủ/thị trường, dự báo tương lai, câu hỏi "tại sao/có nên", lương/chấm công, thuộc tính sản phẩm, vị trí kệ. Một câu chỉ bị gắn unanswerable khi **đồng thời**: (1) có từ khoá chủ đề (theo ranh giới từ; cụm nhiều từ so cả dạng không dấu, từ đơn chỉ so có dấu để tránh "bão" ~ "bao"); (2) shop **không có** cột/bảng tương ứng; (3) từ khoá **không phải giá trị trong dữ liệu** (vd nhóm hàng "Dị ứng"); (4) từ khoá **không phải thuật ngữ do rulebook đang dùng định nghĩa** (vd "chăm chỉ", "trung thành" có định nghĩa tính được trong rulebook public nên không bao giờ bị gắn unanswerable). Điều kiện (4) tự áp dụng cho mọi rulebook, kể cả rulebook mới. Đo lại trên train: vẫn 47 đúng / 0 báo nhầm sau khi mở rộng (mở rộng không làm tăng báo nhầm); trên public: 10/1000 câu bị gắn (trước 8), chỉ đếm số lượng, không dùng để chỉnh luật.

---

## 15. Từ điển ngôn ngữ - dữ liệu (`assets/lexicon.json`)

Từ điển học từ **bộ train** và được nạp vào pipeline ở cả phía câu hỏi lẫn phía dữ liệu (`src/lexicon.py`). **Tuyệt đối không dùng public_test/private_test**: `tools/mine_lexicon.py` chỉ chạy khi thư mục có `answers.jsonl` và đường dẫn không chứa "public"/"private" (có test kiểm tra).

| Phần | Dùng ở đâu | Nội dung |
| :--- | :--- | :--- |
| `query_abbreviations` | `VietnameseNormalizer` (đọc hiểu câu hỏi) | viết tắt/tiếng lóng: dthu, bn, bnhiu, hnay, hqua, trc, vs, ko, hok, nhìu, gium...; có điều kiện theo từ đứng trước/sau: "th trước"→"tháng trước", "bằng tm/ck"→"tiền mặt/chuyển khoản", "mức ck"→"chiết khấu"; "dc 2" (mã) và chuỗi trong ngoặc kép không bị đổi. Bỏ các viết tắt mơ hồ không có bằng chứng ("dt", "tk", "ck" trơ trụi) |
| `value_concepts` | `build_value_hints` → prompt (nối câu hỏi với dữ liệu) | cách nói ↔ giá trị mã hoá trong dữ liệu: "tiền mặt"↔`cash`, "ví điện tử"↔`ewallet`, "tại quầy"↔`store`, "zalo OA"↔`zalo_oa`, "điện thoại"↔`phone`... Chỉ gợi ý khi tìm thấy giá trị đó trong một cột phân loại thật của shop, kèm tên bảng/cột |
| `status_variants` | `normalize_order_status` (nạp dữ liệu) | biến thể trạng thái đơn → 4 trạng thái chuẩn |
| `column_roles` | bảng đồng nghĩa tên cột khi nạp dữ liệu | tên cột theo vai trò, nối thêm vào danh sách có sẵn (không đổi thứ tự ưu tiên cũ) |
| `unanswerable_topics` | `RuleEngine.detect_unanswerable_deterministic` | chủ đề "dữ liệu không có thông tin này" (mục 14, #50), chuyển từ code sang từ điển |

**Kiểm chứng (chỉ trên train, `python tools/mine_lexicon.py --train-root <train>`)**
- Mỗi viết tắt có bằng chứng số câu train chứa nó (dthu 482, bn 211, bnhiu 184, đc/dc 229, hnay 60, trc 60, hqua 48...). Mục không có bằng chứng riêng (sl, đvt, hsd, kh...) giữ vì là viết tắt thông dụng chung, và tool in danh sách đó để biết.
- Kiểm chứng chéo theo shop (chia đôi 30 shop): ở nửa B, 22/25 giá trị cột trạng thái được nhận dạng, 3 giá trị còn lại không phải trạng thái đơn ("Mã đơn hàng", "Đang hoạt động", "Đã nhập hàng"); các giá trị chỉ có ở B (không có ở A) được nhận dạng 13/13. Khái niệm giá trị mã hoá có mặt ở cả hai nửa shop (vd tiền mặt: 5 shop ở A, 2 shop ở B).
- Đo lại trên train sau khi chuyển sang từ điển: unanswerable vẫn 47 đúng / 0 báo nhầm; gợi ý kiểu đáp án đúng 2914/2930 (99.5%) câu có đáp án.
- Hạn chế: từ điển chỉ phủ những gì xuất hiện ở train. Cách nói/ mã giá trị lạ ở tập khác sẽ không được nối; khi đó pipeline chạy như cũ (không hỏng), chỉ không được gợi ý.

---

## 16. Tối ưu tốc độ phía mã nguồn (không đánh đổi độ chính xác)

Mọi tối ưu dưới đây chỉ bỏ tính toán lặp lại; kết quả được so với bản gốc trên dữ liệu train thật.

| Thay đổi | Trước | Sau | Kiểm chứng kết quả giống nhau |
| :--- | :--- | :--- | :--- |
| Gợi ý thực thể (`find_grounding_hints`): tiền kiểm tra chuỗi con trước regex, cache danh sách khoá đã sắp xếp và từ của khoá theo shop, dựng chỉ mục bằng `zip` thay `iterrows` | 274 s / 2881 câu (~95 ms/câu) | 12.9 s / 3000 câu (~4 ms/câu) | 2881/2881 câu cho gợi ý **giống hệt** bản gốc |
| Schema prompt: cache tập cột hiển thị và khối mô tả theo bảng | 0.4–0.5 s/câu trên shop lớn | ~0 sau câu đầu mỗi shop | cùng văn bản |
| Khoảng ngày dữ liệu của shop: tính một lần mỗi shop | ~0.5 s/câu | ~0 | cùng kết quả |
| Từ khoá unanswerable/từ điển: biên dịch regex một lần, cache `strip_accents` (trước đây biên dịch lại ~600 regex/câu, tràn cache của Python) | — | nhanh hơn rõ rệt | unanswerable vẫn 47 đúng / 0 báo nhầm; 2953 + 47 quyết định giống hệt trước |
| Chỉ mục cột phân loại cho gợi ý giá trị mã hoá: tính một lần mỗi shop | quét lại mọi cột mỗi câu | tra từ điển | 92/92 câu cho gợi ý giống hệt thuật toán gốc |
| Nạp dữ liệu song song (`--ingest-workers`, mặc định 4): các shop chưa có cache được nạp bằng nhiều tiến trình rồi ghi Parquet | 1–60 s/shop tuần tự | chia theo số tiến trình | So bản nạp trực tiếp với bản đọc từ Parquet trên 3 shop thật: giá trị giống hệt. Lỗi ở tiến trình con tự quay về nạp tuần tự |

Toàn bộ 3000 câu train qua các bước trước LLM: trung bình **0.18 s/câu** (đã gồm 1–5 s khởi động một lần mỗi shop), không câu nào lỗi, so với 0.22 s/câu trước đó.

**Lỗi tìm thấy khi đo tốc độ**: ở shop có cột tên hàng đồng thời là cột đơn vị (vd "Hàng - Unit"), bước dựng chỉ mục thực thể tạo ra DataFrame trùng cột và **làm lỗi mọi câu hỏi của shop đó** (119 câu train của một shop nhận abstain vì `exception_before_prompt`). Đã sửa: không dùng chính cột tên làm cột ĐVT. Đây là cải thiện độ chính xác, không phải đánh đổi.

---

## 17. Đọc dữ liệu đã preprocess (`<tập>_preprocessed/`)

Bố cục được hỗ trợ: `<tập>_preprocessed/{rulebook.md, calendar.csv, questions.jsonl, <shop>/{tables/*.csv, _readme.txt, ...}}` (shop nằm thẳng trong thư mục tập, không có `data/`).

```bash
python run_pipeline.py \
    --questions "<...>/public_test_preprocessed/questions.jsonl" \
    --data-root "<...>/public_test_preprocessed" \
    --cache-root /content/cache      # tuỳ chọn: nơi ghi cache Parquet (nên là thư mục có quyền ghi)
```

- Rulebook/calendar được tìm ngay trong `--data-root` (đã thử: nạp đúng "bộ public test"). README của shop đọc từ `_readme.txt`.
- Nhánh đọc `tables/*.csv` nay thực hiện thêm:
  1. **Bỏ bảng gộp trùng** (`_drop_redundant_union_tables`): dữ liệu preprocessed bản cũ chứa cả bảng gốc theo tháng lẫn bảng gộp (vd bảng cả năm, bảng theo tháng gộp hai năm). Giữ các bảng gốc mịn nhất và bỏ bảng gộp, nhận diện theo **nội dung** (băm các cột số/ngày/mã, ≥ 99% dòng của bảng nhỏ nằm trong bảng lớn và các bảng nhỏ đủ tổng số dòng để chứa bảng lớn), không theo tên bảng. Cần vì bảng gộp cũ có thể sai (ở một shop có thêm 305 dòng trùng) và nếu giữ cả hai thì bị đếm đôi. Bảng danh mục không bị nhầm (không cùng cấu trúc).
  2. **Gộp các bảng theo kỳ** còn lại thành một bảng (như nhánh đọc dữ liệu gốc); bảng gộp lấy lại tên của bảng gộp cũ đã bỏ.
  3. **Chạy lại Mục 8** (bỏ dòng trùng của cùng một đơn, dòng tổng) cho nguồn csv/xlsx, trừ shop mà README liệt kê toàn nguồn `sqlite` (giống nhánh đọc dữ liệu gốc). Bản preprocessed cũ chưa bỏ các dòng trùng này.
- `--cache-root` mới; mặc định cache nằm ở `.cache/ingested` cạnh thư mục tập (bố cục `<tập>/data/` thì ở thư mục cha của tập như trước).

**Đối chiếu với nhánh đọc dữ liệu gốc (14 shop public, số dòng bảng bán hàng chính)**: giống hệt ở 12/14 shop (vd 90297, 84146, 75970, 119070, 22490, 10131). Hai shop khác nhau:
- Shop có sổ bán hàng 18 sheet theo tháng: nhánh preprocessed ra 32914 dòng, nhánh đọc dữ liệu gốc chỉ 15382 (nguồn có 33493 dòng thô). **Nhánh đọc dữ liệu gốc đang mất hơn nửa dữ liệu ở shop này**: cột mã đơn tên "Đơn - Ma don" (tiêu đề nhóm cột có tiền tố "Đơn - ") không được nhận là cột mã đơn, nên sau khi ép kiểu số, các dòng tiếp theo của đơn (khách/mã đơn để trống theo luật fill-down) bị coi là dòng tổng và bị xoá. Lỗi này có từ bản gốc, chưa sửa (preprocessed không bị).
- Một shop khác lệch 587 dòng giữa hai nhánh (153326 so với 152739), chưa tìm nguyên nhân.

---

## 18. Chạy trên Google Colab (`notebooks/zaloAI_colab.ipynb`)

Dữ liệu trên Drive: `MyDrive/zaloAI/data/{train_preprocessed_full, public_test_preprocessed}`; kết quả lưu vào `MyDrive/zaloAI/ketqua/<tên lần chạy>/`.

| Cell | Việc | Kết quả (Drive) |
| :--- | :--- | :--- |
| 1 | Mount Drive, `git pull` code, chép dữ liệu vào SSD (bỏ `.duckdb`, `samples/`), khai báo hàm `run_split` | — |
| 2 | Bật vLLM (Qwen2.5-Coder-14B-AWQ, `--max-model-len 16384`, prefix caching) | — |
| 3 | 300 câu train mẫu, có đối chiếu đáp án | `ketqua/train300/` |
| 4 | 3000 câu train, có đối chiếu đáp án | `ketqua/train3000/` |
| 5 | 1000 câu public test, không đối chiếu | `ketqua/public1000/` |
| 6 | Xem log vLLM / GPU (tuỳ chọn) | — |
| 7 | Tắt vLLM, tuỳ chọn ngắt runtime | — |

Mỗi lần chạy sinh:
- `submission.jsonl` — file nộp bài, mỗi dòng `{"id", "type", "answer"}` (định dạng của cuộc thi là JSON Lines);
- `detail.json` — mảng JSON, mỗi câu mô tả quyết định của **Prompt 1** và **Prompt 2** (trạng thái, đáp án, mã đã sinh, kết quả chạy), Prompt 3 nếu có bỏ phiếu, lý do chốt đáp án bằng tiếng Việt (`decision_summary`), và với tập train có thêm `gold`, `verdict` (ĐÚNG/SAI/BỎ QUA) và `score`;
- `evaluation.json` — (chỉ train) số đúng/sai/abstain, PA, thống kê theo kiểu đáp án, các câu sai mẫu;
- `run.log` — toàn bộ log. Log hiển thị trực tiếp trong cell, mỗi 5 câu một dòng; với tập train có thêm điểm PA tạm thời (đúng/sai/bỏ qua) cập nhật liên tục, và cuối lần chạy in bảng kết quả theo từng kiểu đáp án cùng vài câu sai để xem nguyên nhân.

Tham số mới của `run_pipeline.py` cho việc này: `--detail-json`, `--eval-json`, `--log-every`, `--cache-root`.
