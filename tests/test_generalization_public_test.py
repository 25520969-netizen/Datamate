"""
Kiểm thử tính khái quát hóa (Generalization Test) trên tập dữ liệu hoàn toàn mới (public_test / dev_*).
Đảm bảo hệ thống KHÔNG overfit vào tập train và có thể tự động nạp, suy luận schema,
tạo kết nối DuckDB trên bất kỳ doanh nghiệp unseen nào mà không cần can thiệp thủ công.
"""

from pathlib import Path
import pytest
from src.ingest import BusinessDataStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DATA_ROOT = PROJECT_ROOT / "public_test" / "data"


@pytest.mark.skipif(not PUBLIC_DATA_ROOT.exists(), reason="public_test/data không tồn tại cục bộ")
def test_generalization_public_test_ingestion():
    store = BusinessDataStore(PUBLIC_DATA_ROOT)
    
    # Kiểm tra 3 shop đại diện với các loại cấu trúc khác nhau trong public_test:
    # dev_001: Excel chi tiết đơn + danh mục hàng
    # dev_002: Excel phân mảnh theo từng tháng (T1_2025 -> T5_2026) + quy đổi
    # dev_003: Excel tổng hợp theo năm (2025 + 2026) + phiếu thu + tồn kho csv
    for dev_id in ["dev_001", "dev_002", "dev_003"]:
        tables = store.load_tables(dev_id, use_cache=True)
        assert len(tables) > 0, f"{dev_id} không nạp được bảng nào!"
        
        # Kiểm tra DuckDB connection khởi tạo thành công
        con, registered_tables = store.create_duckdb_connection(dev_id)
        assert len(registered_tables) == len(tables)
        
        # Đảm bảo có thể thực thi truy vấn DuckDB cơ bản mà không crash
        duck_tables = con.execute("SHOW TABLES;").fetchall()
        assert len(duck_tables) > 0
        
        # Kiểm tra Schema Compact Prompt sinh ra hợp lệ và không rỗng
        schema_prompt = store.build_compact_schema_prompt(dev_id)
        assert len(schema_prompt) > 50
        assert "[Bảng:" in schema_prompt
