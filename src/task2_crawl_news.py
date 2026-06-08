"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.

Cài đặt: pip install crawl4ai && crawl4ai-setup
"""
import asyncio
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    # Châu Việt Cường — phê ma tuý rồi sát hại cô gái 2018-2019
    "https://vnexpress.net/ca-si-chau-viet-cuong-nhan-13-nam-tu-vi-nhet-toi-hai-chet-co-gai-3891028.html",
    "https://vnexpress.net/ca-si-chau-viet-cuong-hau-toa-vi-nhet-toi-hai-chet-co-gai-20-tuoi-3890738.html",
    # Phúc XO — bị bắt vì tổ chức sử dụng ma tuý 2018
    "https://vnexpress.net/phuc-xo-va-em-trai-bi-bat-khan-cap-3908877.html",
    "https://vnexpress.net/phap-luat/phuc-xo-va-11-nguoi-bi-bat-tam-giam-3913543.html",
    # Miu Lê — bị bắt quả tang dùng ma tuý ở Cát Bà 2024
    "https://vnexpress.net/ca-si-miu-le-bi-bat-qua-tang-dung-ma-tuy-o-bai-bien-5072657.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    # Tổng hợp nhiều nghệ sĩ
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    # Long Nhật & Sơn Ngọc Minh bị bắt 2026
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    # Nhiều nghệ sĩ nổi tiếng bị khởi tố — tổng kết cuối 2025
    "https://tuoitre.vn/vien-kiem-sat-tp-hcm-nhieu-nghe-si-nguoi-noi-tieng-bi-khoi-to-do-lien-quan-ma-tuy-20251209142132042.htm",
]


def _slugify(text: str, max_len: int = 70) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].rstrip("-")


def _extract_title(result, url: str) -> str:
    if result.metadata and result.metadata.get("title"):
        return result.metadata["title"].strip()
    for line in (result.markdown or "").splitlines():
        line = line.lstrip("#").strip()
        if len(line) > 10:
            return line
    # fallback: slug từ URL
    return urlparse(url).path.split("/")[-1].replace(".html", "").replace("-", " ")


async def crawl_article(crawler: AsyncWebCrawler, url: str) -> dict | None:
    try:
        result = await crawler.arun(url=url)
        if not result.success:
            print(f"  ✗ Thất bại: {url}")
            return None

        content = str(result.markdown) if result.markdown else ""
        if len(content) < 200:
            print(f"  ✗ Nội dung quá ngắn ({len(content)} chars): {url}")
            return None

        return {
            "url": url,
            "title": _extract_title(result, url),
            "crawl_date": datetime.now().isoformat(),
            "source_domain": urlparse(url).netloc,
            "content_markdown": content,
            "content_length": len(content),
        }
    except Exception as e:
        print(f"  ✗ Lỗi {url}: {e}")
        return None


def save_article(data: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(data["title"]) or _slugify(data["url"].split("/")[-1])
    path = output_dir / f"{slug}.json"
    counter = 1
    while path.exists():
        path = output_dir / f"{slug}-{counter}.json"
        counter += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def crawl_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    success, fail = 0, 0

    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] {url}")
            article = await crawl_article(crawler, url)
            if article:
                path = save_article(article, DATA_DIR)
                print(f"  ✓ Lưu: {path.name} ({article['content_length']} chars)")
                success += 1
            else:
                fail += 1

    print(f"\n=== Kết quả: {success} thành công / {fail} thất bại ===")
    print(f"Output: {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
