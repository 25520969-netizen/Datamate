# -*- coding: utf-8 -*-
"""
Module 3: Code Generation Engine (Qwen2.5-Coder-7B-Instruct) & End-to-End Multi-Stage Agent.
Kết hợp:
  - Module 1 (`BusinessDataStore` trong `src/ingest.py`)
  - Module 2 (`DateResolver` trong `src/date_resolver.py` & `RuleEngine` trong `src/rule_engine.py`)
  - Module 3 (`QwenCoderClient` sử dụng `Qwen/Qwen2.5-Coder-7B-Instruct`)
  - Module 4 (`PythonSandbox` trong `src/sandbox.py` với vòng lặp tự sửa lỗi tối đa 2 lần)
  - Module 5 (`postprocess_and_guard` trong `src/postprocess.py` bảo vệ điểm Penalized Accuracy)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from src.date_resolver import DateResolver
from src.ingest import BusinessDataStore
from src.postprocess import postprocess_and_guard
from src.rule_engine import DAILFewShotSelector, DynamicEntityGrounder, RuleEngine
from src.sandbox import PythonSandbox


DEFAULT_QWEN_7B_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OLLAMA_7B_MODEL = "qwen2.5-coder:7b"

SYSTEM_PROMPT = """Bạn là Chuyên gia Khoa học Dữ liệu giải quyết bài toán Hỏi đáp trên dữ liệu bảng doanh nghiệp (DataMate - Zalo AI Challenge 2026).

NHIỆM VỤ:
Viết một khối mã Python hoàn chỉnh, chính xác tuyệt đối để truy vấn dữ liệu từ biến `tables` (kiểu `dict[str, pd.DataFrame]`) hoặc `con` (DuckDB in-memory đã nạp đủ bảng) nhằm trả lời câu hỏi.

MÔI TRƯỜNG SANDBOX:
- `tables`: dict chứa các bảng DataFrame (ngày đã chuẩn hóa 'YYYY-MM-DD', tiền/số lượng là số, trạng thái đơn chuẩn 4 giá trị: 'Hoàn thành', 'Đã hủy', 'Trả hàng', 'Đang giao').
- `con`: kết nối DuckDB in-memory đã đăng ký toàn bộ bảng trong `tables`.
- `as_of_date`: chuỗi ngày 'YYYY-MM-DD' (hôm nay).
- `date_context`: dict mốc thời gian tính sẵn từ calendar (vd: date_context['this_month'] -> ('2025-03-01', '2025-03-18')).
- Thư viện sẵn có: pd, np, duckdb, re, math, date, datetime, timedelta, unicodedata.

QUY TẮC CỐT LÕI:
1. PHÂN BIỆT SỐ LƯỢNG HÀNG vs DOANH THU & XẾP HẠNG BÁN CHẠY:
   - Khi câu hỏi hỏi bán chạy nhất / bán nhiều nhất / hot nhất: BẮT BUỘC THEO SỐ LƯỢNG QUY VỀ ĐƠN VỊ CƠ BẢN (Mục 3 Rulebook). ƯU TIÊN SỐ 1 CỘT ĐƠN VỊ CƠ BẢN ('qty_base', 'SL lẻ', 'Quy ra ĐV lẻ', 'Số lượng lẻ', 'SL cơ bản') nếu bảng có cột này, vì các mặt hàng khác nhau về quy cách đóng gói (thùng/lon/chai). BẮT BUỘC dùng `ORDER BY SUM(COALESCE("qty_base", "Số lượng")) DESC`. TUYỆT ĐỐI KHÔNG xếp hạng bán chạy theo doanh thu / thành tiền!
   - Khi câu hỏi hỏi về MỘT MÓN ĂN / MẶT HÀNG CỤ THỂ (hoặc có gợi ý thực thể): BẮT BUỘC PHẢI THÊM ĐIỀU KIỆN LỌC TÊN MÓN TRONG WHERE (ví dụ: `WHERE "Tên hàng" ILIKE '%cơm chiên dương châu%'`). TUYỆT ĐỐI KHÔNG tính tổng cả tháng của toàn cửa hàng!
   - Khi câu hỏi hỏi số lượng cụ thể: 'bao nhiêu nồi', 'bao nhiêu đĩa', 'mấy đĩa', 'mấy nồi', 'bao nhiêu kg', 'may kg', 'số kg', 'so kg', 'số lượng kg', 'mấy lon', 'bao nhiêu lon', 'số lon', 'mấy gói', 'bao nhiêu gói', 'số gói', 'bao nhiêu chai', 'mấy chai', 'số chai', 'bao nhiêu hộp', 'mấy hộp', 'số hộp', 'bao nhiêu chiếc', 'mấy chiếc', 'số chiếc', 'bao nhiêu cái', 'mấy cái', 'số cái', 'bao nhiêu ly', 'mấy ly', 'số ly', 'bao nhiêu bao', 'mấy bao', 'số bao', 'bao nhiêu vỉ', 'mấy vỉ', 'số vỉ', 'bao nhiêu viên', 'mấy viên', 'số viên', 'bao nhiêu tuýp', 'mấy tuýp', 'số tuýp', 'bao nhiêu thùng', 'mấy thùng', 'số thùng', 'bao nhiêu lốc', 'mấy lốc', 'số lốc', 'bao nhiêu lít', 'mấy lít', 'số lít', 'số lượng', 'lượng bán': BẮT BUỘC TÍNH TỔNG TRỰC TIẾP TỪ CỘT SỐ LƯỢNG ('Số lượng', 'SoLuong', 'SL bán', 'so_luong', 'Số lượng sản phẩm', 'Số lượng bán') CỦA MẶT HÀNG ĐÓ. TUYỆT ĐỐI KHÔNG tự ý nhân thêm bảng quy đổi nếu câu hỏi đã hỏi đúng đơn vị ghi nhận trong bảng (vd: hỏi bao nhiêu kg mà đơn vị bán đã là kg thì sum trực tiếp cột số lượng)! TUYỆT ĐỐI KHÔNG TÍNH TỔNG TIỀN / THÀNH TIỀN KHI HỎI SỐ LƯỢNG!
   - Khi câu hỏi hỏi tiền / doanh thu: 'bao nhiêu tiền', 'mấy đồng', 'doanh thu', 'dthu', 'tiền hàng', 'thu về bao nhiêu', 'trị giá bao nhiêu': ƯU TIÊN SỐ 1 DÙNG CỘT TIỀN HÀNG THUẦN ('revenue_norm', 'Tien hang', 'Tiền hàng', 'Thành tiền', 'Giá trị hàng', 'thanh_tien', 'Amount', 'Thành tiền (đ)'). TUYỆT ĐỐI KHÔNG dùng cột 'TongTien', 'Tổng sau giảm giá' hay 'Khách đã trả' (vì các cột này bị cộng thêm phụ phí, phí vận chuyển hoặc tiền tip).
   - Lọc đơn hợp lệ (bắt buộc khi tính doanh thu, số đơn, số lượng bán, bán chạy nhất):
     + Trong pandas: `df[~df['Trạng thái'].astype(str).str.lower().str.contains('hủy|huỷ|trả|cancel|return', regex=True)]`
     + Trong DuckDB SQL: `WHERE ("Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%huỷ%' AND "Trạng thái" NOT ILIKE '%trả%' AND "Trạng thái" NOT ILIKE '%cancel%' AND "Trạng thái" NOT ILIKE '%return%')` (hoặc nếu có cột status_norm: `WHERE "status_norm" = 'Hoàn thành'`).
     TUYỆT ĐỐI KHÔNG dùng từ 'hoàn' để lọc vì sẽ lọc mất 'Hoàn thành' / 'Đã hoàn thành' (đây là đơn bán thành công hợp lệ!). Các đơn 'Hoàn thành', 'Đã hoàn thành', 'Thành công', 'xong', 'đã lấy', 'completed' BẮT BUỘC PHẢI GIỮ LẠI.
   - Lưu ý Mục 3 Rulebook: Mặt hàng dịch vụ 'Phí phục vụ' KHÔNG PHẢI MẶT HÀNG BÁN và bắt buộc loại trừ khi xếp hạng mặt hàng bán chạy / doanh thu / số lượng.
2. TỈ LỆ ĐƠN HỦY VÀ PHẦN TRĂM:
   - Tỉ lệ đơn hủy (Cancellation Rate): KHÔNG lọc bỏ đơn hủy!
     + Trong DuckDB SQL: BẮT BUỘC DÙNG `SELECT ROUND(COUNT(DISTINCT CASE WHEN "status_norm" = 'Đã hủy' THEN "cột_mã_đơn" END) * 100.0 / NULLIF(COUNT(DISTINCT "cột_mã_đơn"), 0), 2)`. TUYỆT ĐỐI KHÔNG DÙNG `COUNT(*)` đếm số dòng sản phẩm!
     + Trong pandas: `order_status = df.groupby(cột_mã_đơn)['Trạng thái'].first(); cancel_rate = (order_status.astype(str).str.lower().str.contains('hủy|huỷ')).sum() / len(order_status) * 100`
   - Phân biệt các khái niệm so sánh thời gian:
     + 'So với tháng liền trước' / 'so với tháng trước': So với CẢ THÁNG M-1 (ví dụ: T6/2026 so với cả T5/2026, TUYỆT ĐỐI KHÔNG so với cùng kỳ năm trước T6/2025).
     + 'Cùng kỳ tháng trước': Từ ngày 1 đến cùng ngày D của tháng M-1 (ví dụ: 01/07/2026-15/07/2026 so với 01/06/2026-15/06/2026).
     + 'Cùng kỳ năm trước': So với cùng tháng M của năm Y-1 (ví dụ: T5/2026 so với T5/2025).
   - Định dạng percent: BẮT BUỘC LÀ SỐ ĐÃ NHÂN 100 (ví dụ 40.3 chứ TUYỆT ĐỐI KHÔNG để 0.403; 35.17 chứ không phải 0.3517). Result: `{"type": "percent", "answer": 40.3}`.
   - Tỉ trọng / Đóng góp: Tỉ lệ của một phần so với toàn bộ không bao giờ vượt quá 100% (nằm trong khoảng 0% đến 100%).
3. Xếp hạng khách hàng: Đơn không có mã hoặc ghi 'Khách lẻ' / 'lẻ' / 'kl' KHÔNG PHẢI KHÁCH và không tính vào xếp hạng khách hàng. Bắt buộc loại bỏ:
   - Trong pandas: `df[~df[cột_khách].astype(str).str.lower().isin(['khách lẻ', 'khach le', 'lẻ', 'le', 'kl', 'none', 'nan'])]`.
   - Trong DuckDB SQL: `WHERE "Tên khách" NOT ILIKE '%khách lẻ%' AND "Tên khách" NOT ILIKE '%lẻ%' AND "Tên khách" NOT ILIKE '%kl%' AND "Tên khách" IS NOT NULL`.
4. Đếm số đơn (bán được bao nhiêu đơn, số đơn, số bill, số phiếu):
   - Khi bảng là chi tiết đơn hàng (có nhiều dòng sản phẩm trên cùng 1 đơn, có cột mã đơn/mã hóa đơn/mã phiếu/order id):
     BẮT BUỘC DÙNG `SELECT COUNT(DISTINCT "cột_mã_đơn")` (trong pandas: `df['cột_mã_đơn'].nunique()`),
     TUYỆT ĐỐI KHÔNG DÙNG `COUNT(*)` hay `len(df)` vì sẽ bị đếm nhầm số dòng sản phẩm thay vì số đơn!
   - BẮT BUỘC kết hợp lọc đơn hợp lệ: bỏ các đơn bị hủy/trả hàng: `AND ("Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%trả%')`.
   - Khi hỏi số đơn có doanh thu >= X: groupby mã đơn tính tổng tiền trước, lọc >= X rồi đếm số đơn duy nhất.
5. Lọc thời gian: ƯU TIÊN SỐ 1 là lọc đúng khoảng ngày được chỉ định trong câu hỏi hoặc mốc ⭐ ở đầu ngữ cảnh thời gian (vd: 'hôm nay', 'hôm qua', 'tháng 8/2025', 'quý 2/2025').
   - Tất cả các bảng dữ liệu trong `tables` ĐÃ ĐƯỢC CHUẨN HÓA ngày về dạng chuỗi ISO 'YYYY-MM-DD' và có sẵn alias cột 'Ngày'.
   - Cách lọc chuẩn xác và an toàn nhất:
     + Trong pandas: `(df['Ngày'].astype(str).str.slice(0, 10) >= start) & (df['Ngày'].astype(str).str.slice(0, 10) <= end)`.
     + Trong DuckDB: `WHERE CAST("Ngày" AS VARCHAR) >= 'start' AND CAST("Ngày" AS VARCHAR) <= 'end'` (hoặc `WHERE CAST("Ngày" AS DATE) BETWEEN DATE 'start' AND DATE 'end'`).
     + TUYỆT ĐỐI KHÔNG dùng cú pháp định dạng `%d/%m/%Y` vì dữ liệu ngày trong sandbox đã là chuẩn ISO 'YYYY-MM-DD'!
6. So khớp chuỗi: Lowercase cả hai vế: `df[col].astype(str).str.strip().str.lower() == target_str.lower()`. Thuốc kê đơn / Hoạt chất: so sánh bằng chính xác `==` (không dùng `.str.contains()`).
7. Không trả lời được: Khi câu hỏi hỏi thông tin mà shop không ghi nhận (địa chỉ/tên đường khi shop không có cột địa chỉ; giới tính khách khi không có cột giới tính; đánh giá cảm tính 'tận tâm', 'chu đáo', 'lễ phép'; liều dùng y khoa; lý do hủy đơn; lãi gộp khi không có giá vốn; tiền nhập khi không có dữ liệu nhập; tồn kho khi shop không có bảng tồn kho / nhập hàng; hoặc kỳ hỏi hoàn toàn nằm ngoài khoảng ngày có dữ liệu của shop): BẮT BUỘC TRẢ VỀ: `result = {"type": "unanswerable", "reason": "Dữ liệu không có thông tin này"}`.
8. CÔNG NỢ, KÊNH BÁN, PHƯƠNG THỨC THANH TOÁN & TỒN KHO:
   - Công nợ / Tiền còn nợ:
     + Nếu có cột 'Chưa trả', 'Còn nợ', 'Dư nợ': lấy trực tiếp từ cột này.
     + Nếu có bảng đơn hàng và bảng thanh toán riêng (Mục 4 Rulebook): Nợ của đơn = don_hang.TongThanhToan - sum(thanh_toan.SoTienThu) cho các đơn có PhuongThucTT == 'debt' (hoặc chưa trả đủ).
   - Bảng khách hàng & Tên khách hàng:
     + Nếu bảng bán hàng ĐÃ CÓ cột tên khách (vd: 'Khách hàng', 'Tên khách hàng', 'TenKH', 'Bên mua', 'Khách hàng (tên)'), DÙNG TRỰC TIẾP TÊN KHÁCH HÀNG TỪ CỘT ĐÓ, TUYỆT ĐỐI KHÔNG join bảng khác!
     + Chỉ join sang bảng khách hàng (như DanhSachKhachHang, BanHang_KhachHang, khach_hang) khi bảng bán hàng CHỈ CÓ mã khách (như Customer ID, Mã KH) mà không có tên khách.
     + TUYỆT ĐỐI KHÔNG để mã định danh KHxxxx làm đáp án tên khách hàng khi có tên khách!
   - Kênh bán hàng: nhận diện linh hoạt các cột như 'Bán qua', 'Kênh', 'Nguồn', 'Hình thức bán' (ví dụ: store, zalo_oa, phone, grab, shopeefood).
   - Phương thức thanh toán: nhận diện linh hoạt các cột như 'Trả bằng', 'HTTT', 'PTTT', 'Phương thức thanh toán' (ví dụ: tiền mặt, chuyển khoản, cod, công nợ).
   - Tồn kho: Nếu có bảng nhập hàng và bán hàng chi tiết, Tồn kho = Tổng số lượng nhập - Tổng số lượng bán của sản phẩm.
9. CÚ PHÁP DUCKDB SQL:
   - Khi dùng `con.execute(sql)`, TẤT CẢ tên cột có dấu cách, tiếng Việt có dấu hoặc ký tự ngoặc `()` BẮT BUỘC PHẢI BỌC TRONG DẤU NGOẶC KÉP, ví dụ: `"Khách hàng (tên)"`, `"Mã đơn hàng"`, `"Số lượng"`, `"Còn nợ"`, `"Đơn giá (đ)"`. TUYỆT ĐỐI KHÔNG tự ý thay dấu cách thành gạch dưới `_` hay viết `Khách_hàng_(tên)`.
   - Mẫu chuẩn cho các dạng câu hỏi phổ biến:
     + Doanh thu: `SELECT SUM("Thành tiền") FROM ... WHERE CAST("Ngày" AS VARCHAR) BETWEEN '...' AND '...' AND ("Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%trả%')`
     + Số đơn: `SELECT COUNT(DISTINCT "Mã đơn hàng") FROM ... WHERE CAST("Ngày" AS VARCHAR) BETWEEN '...' AND '...' AND ("Trạng thái" NOT ILIKE '%hủy%' AND "Trạng thái" NOT ILIKE '%trả%')`
     + Ngày cao nhất (type: date): `SELECT "Ngày", SUM(...) as metric FROM ... GROUP BY "Ngày" ORDER BY metric DESC LIMIT 1` -> Trích xuất `result = {"type": "date", "answer": str(row[0])[:10]}` (BẮT BUỘC LẤY CỘT NGÀY `row[0]`, TUYỆT ĐỐI KHÔNG LẤY CỘT SỐ LIỆU `row[1]`!).
10. ĐỊNH DẠNG ĐÁP ÁN CUỐI CÙNG (`result`):
   - `string`: CHỈ LÀ 1 CHUỖI TÊN ĐỐI TƯỢNG (vd: "Trà đào cam sả (M)"), TUYỆT ĐỐI KHÔNG ghép thêm số lượng, số đếm hay đơn vị, KHÔNG ghép nhiều tên bằng dấu phẩy! BẮT BUỘC giữ nguyên hậu tố size như '(M)', '(L)' nếu có trong dữ liệu gốc.
   - `list`: Mảng JSON các phần tử `["A", "B", ...]` hoặc `[val1, val2, ...]`. Khi hỏi 'mỗi loại bao nhiêu', 'mỗi bên bao nhiêu', 'lần lượt là bao nhiêu', 'mấy hộp chẵn, mấy gói lẻ', 'tuổi nợ', hoặc top N đối tượng: BẮT BUỘC trả về list mảng JSON theo đúng thứ tự câu hỏi yêu cầu. Giữ nguyên hậu tố size nếu tên sản phẩm gốc có size (vd: 'Trà đào (L)').
   - `table`: Khi hỏi Top N KÈM số tiền/doanh thu/công nợ (hoặc hỏi mỗi người/từng người còn nợ bao nhiêu): bắt buộc trả về `{"columns": ["khach_hang", "so_tien_no"], "rows": [["Tên Khách A", 1000], ...]}`. Lưu ý: cột khách hàng phải là Tên khách hàng (chuỗi tên người), không phải mã KHxxxx.
   - `number`, `percent`, `boolean`: `{"type": "<kiểu>", "answer": <giá_trị>}` (với percent: answer là số thực ví dụ 40.3 chứ KHÔNG PHẢI 0.403).
   - `date`: BẮT BUỘC TRẢ VỀ CHUỖI NGÀY 'YYYY-MM-DD' (vd: {"type": "date", "answer": "2026-03-26"}), TUYỆT ĐỐI KHÔNG TRẢ VỀ SỐ TIỀN / SỐ ĐẾM (không trả về 269500 hay 3519500)!
   - `unanswerable`: `{"type": "unanswerable", "reason": "..."}`.
11. Chỉ trả về một khối mã ```python ... ``` duy nhất, không giải thích dài dòng."""


class QwenCoderClient:
    """
    Client kết nối tới mô hình Qwen2.5-Coder-7B-Instruct.
    Hỗ trợ 3 chế độ backend:
      - 'openai': Gọi qua OpenAI-compatible API (vLLM, SGLang, LM Studio, Ollama /v1).
      - 'ollama': Gọi trực tiếp qua Ollama API (`/api/chat` với model `qwen2.5-coder:7b`).
      - 'transformers': Nạp trực tiếp `Qwen/Qwen2.5-Coder-7B-Instruct` bằng HuggingFace Transformers (chạy offline trên GPU).
      - 'custom': Sử dụng hàm callback tùy chỉnh (phục vụ kiểm thử / mock).
    """

    def __init__(
        self,
        backend: str = "openai",
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 640,
        request_timeout: float = 90.0,
        custom_generator: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> None:
        self.backend = backend.lower().strip()
        if model_name:
            self.model_name = model_name
        elif self.backend == "ollama":
            self.model_name = DEFAULT_OLLAMA_7B_MODEL
        else:
            self.model_name = DEFAULT_QWEN_7B_MODEL

        self.api_base = (
            api_base
            or os.environ.get("QWEN_API_BASE")
            or ("http://localhost:11434" if self.backend == "ollama" else "http://localhost:8000/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("QWEN_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.custom_generator = custom_generator

        self._hf_tokenizer: Any = None
        self._hf_model: Any = None

    def _ensure_hf_loaded(self) -> None:
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self._hf_model.eval()

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Sinh phản hồi từ danh sách messages theo chuẩn ChatML."""
        if self.custom_generator is not None:
            return self.custom_generator(messages)

        if self.backend == "openai":
            url = f"{self.api_base}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            eff_temp = max(self.temperature, 0.1) if len(messages) > 2 else self.temperature
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": eff_temp,
                "max_tokens": self.max_tokens,
            }
            last_exc: Optional[Exception] = None
            for http_try in range(3):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.request_timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                except requests.RequestException as exc:
                    last_exc = exc
                    if getattr(getattr(exc, "response", None), "status_code", None) == 400:
                        raise
                    if http_try < 2:
                        time.sleep(2.0 * (http_try + 1))
            if last_exc is not None:
                raise last_exc

        if self.backend == "ollama":
            url = f"{self.api_base}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

        if self.backend == "transformers":
            import torch  # type: ignore

            self._ensure_hf_loaded()
            text = self._hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self._hf_tokenizer([text], return_tensors="pt").to(self._hf_model.device)
            with torch.no_grad():
                generated_ids = self._hf_model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                )
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            return self._hf_tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

        raise ValueError(f"Unsupported backend: {self.backend}")


class DataMateAgent:
    """
    Pipeline Multi-Stage Code-Generation Agent hoàn chỉnh (5 Modules):
      1. Ingestion & Schema Standardization
      2. Query Analyzer, Date Resolver & Rule Knowledge Injector
      3. Code Generation Engine (Qwen2.5-Coder-7B-Instruct)
      4. Execution Sandbox & Self-Correction Loop (tối đa 2 lần thử lại)
      5. Post-Processing & Penalized Accuracy Guard (Abstention Policy)
    """

    def __init__(
        self,
        data_root: Path,
        calendar_path: Path,
        rulebook_path: Path,
        llm_client: QwenCoderClient,
        cache_root: Optional[Path] = None,
        sandbox_timeout: float = 10.0,
        max_retries: int = 2,
        verbose: bool = False,
    ) -> None:
        self.store = BusinessDataStore(data_root=data_root, cache_root=cache_root)
        self.date_resolver = DateResolver(calendar_path=calendar_path)
        self.rule_engine = RuleEngine(rulebook_path=rulebook_path)
        self.entity_grounder = DynamicEntityGrounder()
        self.few_shot_selector = DAILFewShotSelector()
        self.sandbox = PythonSandbox(
            data_root=data_root,
            cache_root=cache_root,
            timeout_seconds=sandbox_timeout,
        )
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.verbose = verbose

    def build_user_prompt(
        self,
        question_obj: Dict[str, Any],
        expected_type: str,
        date_prompt: str,
        rules_prompt: str,
        schema_prompt: str,
        entity_hints: str = "",
        few_shot_prompt: str = "",
    ) -> str:
        question = question_obj["question"]
        as_of_date = question_obj["as_of_date"]
        business_id = question_obj["business_id"]

        parts = [
            f"=== CÂU HỎI CẦN GIẢI QUYẾT ===\n"
            f"- Mã câu hỏi: {question_obj.get('id', 'unknown')}\n"
            f"- Mã cửa hàng (`business_id`): {business_id}\n"
            f"- Ngày hiện tại (`as_of_date` - Hôm nay): {as_of_date}\n"
            f"- Nội dung câu hỏi: {question}\n"
            f"- Kiểu đáp án gợi ý (`expected_type`): {expected_type}",
            f"=== NGỮ CẢNH THỜI GIAN ĐÃ QUY ĐỔI QUA CALENDAR.CSV ===\n{date_prompt}",
            schema_prompt,
        ]
        if entity_hints:
            parts.append(entity_hints)
        if few_shot_prompt:
            parts.append(few_shot_prompt)
        parts.extend([
            f"=== LUẬT NGHIỆP VỤ RULEBOOK CẦN ÁP DỤNG ===\n{rules_prompt}",
            "=== YÊU CẦU SINH MÃ PYTHON ===\n"
            "Hãy viết mã Python truy vấn dữ liệu từ `tables` hoặc `con` để tính ra kết quả chính xác.\n"
            f"Cuối đoạn code, bắt buộc gán `result = {{\"type\": \"...\", \"answer\": ...}}` "
            f"(hoặc `{{\"type\": \"unanswerable\", \"reason\": \"...\"}}` / `{{\"type\": \"abstain\"}}`)."
        ])
        return "\n\n".join(parts)

    def answer_question(self, question_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Giải quyết một câu hỏi từ đầu đến cuối (End-to-End).
        Nếu bất kỳ bước nào gặp lỗi không thể phục hồi sau `max_retries` lần tự sửa,
        trả về `{"id": qid, "type": "abstain"}` để bảo toàn điểm số Penalized Accuracy.
        """
        qid = question_obj["id"]
        business_id = question_obj["business_id"]
        question = question_obj["question"]
        as_of_date = question_obj["as_of_date"]

        try:
            # Module 1 & 2: Chuẩn bị Schema, Date Context, Entity Grounding và Rule Knowledge
            biz_readme = self.store.get_readme(business_id)
            tables = self.store.load_tables(business_id)
            entity_hints = self.entity_grounder.find_grounding_hints(question, business_id, tables)
            few_shot_prompt = self.few_shot_selector.select_exemplars(
                question=question,
                has_entity_hint=bool(entity_hints and entity_hints.strip()),
            )
            expected_type = self.rule_engine.infer_expected_type(question)
            date_ctx = self.date_resolver.resolve(question, as_of_date)
            date_prompt = self.date_resolver.format_for_prompt(question, as_of_date)
            rules_prompt = self.rule_engine.select_relevant_rules(question, biz_readme)
            schema_prompt = self.store.build_compact_schema_prompt(business_id, question=question)

            # Kiểm tra câu hỏi không thể trả lời theo tiền định Mục 7 Rulebook
            is_unans, unans_reason = self.rule_engine.detect_unanswerable_deterministic(
                question_obj=question_obj,
                store=self.store,
                date_resolver=self.date_resolver,
            )
            if is_unans:
                return {"id": qid, "type": "unanswerable", "reason": unans_reason}
        except Exception:
            return {"id": qid, "type": "abstain"}

        user_prompt = self.build_user_prompt(
            question_obj=question_obj,
            expected_type=expected_type,
            date_prompt=date_prompt,
            rules_prompt=rules_prompt,
            schema_prompt=schema_prompt,
            entity_hints=entity_hints,
            few_shot_prompt=few_shot_prompt,
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Module 3 & 4: Sinh mã Python & Thực thi trong Sandbox kèm vòng lặp Self-Correction
        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                llm_reply = self.llm_client.generate(messages)
            except Exception as llm_exc:
                resp_body = getattr(getattr(llm_exc, "response", None), "text", "")
                err_detail = f" | {resp_body[:300]}" if resp_body else ""
                print(f"    ⚠️ [LLM Error] {qid} ({business_id}, attempt {attempt+1}): {llm_exc}{err_detail}", flush=True)
                return {"id": qid, "type": "abstain"}

            exec_outcome = self.sandbox.execute(
                code=llm_reply,
                business_id=business_id,
                as_of_date=as_of_date,
                date_context=date_ctx,
            )

            if self.verbose:
                print(
                    f"    [Attempt {attempt+1}/{total_attempts}] {qid} ({business_id}) -> "
                    f"ok={exec_outcome.get('ok')} | result={exec_outcome.get('result')} | err={exec_outcome.get('error')}"
                )

            if exec_outcome.get("ok") and exec_outcome.get("result") is not None:
                # Module 5: Chuẩn hóa định dạng & kiểm tra tính hợp lý (Penalized Accuracy Guard)
                final_pred = postprocess_and_guard(
                    question_id=qid,
                    question=question,
                    raw_output=exec_outcome["result"],
                    expected_type=expected_type,
                )
                return final_pred

            # Nếu thực thi thất bại hoặc trả về None, phản hồi Traceback cho LLM sửa lỗi (nếu còn lượt)
            err_msg = exec_outcome.get("error") or "Biến `result` có giá trị `None` sau khi chạy code."
            if attempt < self.max_retries:
                # Giữ đúng 1 lượt lỗi gần nhất để không vượt quá giới hạn 8192 token của vLLM
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt[:10500]},
                    {"role": "assistant", "content": llm_reply[:1800]},
                    {
                        "role": "user",
                        "content": (
                            f"Đoạn mã Python trên gặp lỗi khi chạy trong Sandbox:\n"
                            f"```text\n{err_msg[:1500]}\n```\n"
                            f"Hãy kiểm tra lại đúng tên bảng trong `tables` và tên cột chính xác đã liệt kê, "
                            f"sau đó chỉ viết lại khối ```python ... ``` đã sửa lỗi và gán `result = {{\"type\": \"...\", \"answer\": ...}}`."
                        ),
                    },
                ]

        # Hết số lượt thử lại mà vẫn lỗi -> kích hoạt Abstention Policy (0 điểm thay vì -1)
        return {"id": qid, "type": "abstain"}
