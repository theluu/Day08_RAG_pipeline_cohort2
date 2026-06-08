"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Nguồn tải thủ công:
    https://thuvienphapluat.vn — tìm theo số hiệu văn bản
"""
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Download file từ URL về DATA_DIR."""
    output_path = DATA_DIR / filename
    if output_path.exists():
        print(f"  Đã có: {filename}")
        return output_path
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  ✓ Đã tải: {filename} ({output_path.stat().st_size // 1024}KB)")
    return output_path


if __name__ == "__main__":
    setup_directory()
    print("\nHướng dẫn tải văn bản pháp luật:")
    print("1. Truy cập: https://thuvienphapluat.vn")
    print("2. Tải 3 văn bản sau vào thư mục:", DATA_DIR)
    print()
    print("  a) Tìm '73/2021/QH15'        → luat-phong-chong-ma-tuy-2021.pdf")
    print("  b) Tìm '105/2021/NĐ-CP'      → nghi-dinh-105-2021.pdf")
    print("  c) Tìm 'Bộ luật Hình sự 2015' → blhs-2015-chuong-xx.pdf")
