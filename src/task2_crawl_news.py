"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.

Cài đặt: pip install crawl4ai && crawl4ai-setup
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Điền URL thực tế của 5 bài báo — tìm trên vnexpress.net, tuoitre.vn, dantri.com.vn
# Từ khoá tìm kiếm: "nghệ sĩ ma tuý", "ca sĩ bị bắt ma tuý", "diễn viên ma tuý"
ARTICLE_URLS = [
    # "https://vnexpress.net/...",
    # "https://vnexpress.net/...",
    # "https://tuoitre.vn/...",
    # "https://dantri.com.vn/...",
    # "https://dantri.com.vn/...",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(crawler: AsyncWebCrawler, url: str) -> dict | None:
    try:
        result = await crawler.arun(url=url)
        if not result.success:
            print(f"  ✗ Thất bại: {url}")
            return None
        return {
            "url": url,
            "title": result.metadata.get("title", ""),
            "date_crawled": datetime.now().isoformat(),
            "content": result.markdown or result.cleaned_html or "",
        }
    except Exception as e:
        print(f"  ✗ Lỗi {url}: {e}")
        return None


async def crawl_all():
    setup_directory()
    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(ARTICLE_URLS):
            if not url or url.startswith("#"):
                continue
            print(f"[{i+1}/{len(ARTICLE_URLS)}] {url}")
            article = await crawl_article(crawler, url)
            if article:
                out = DATA_DIR / f"article_{i+1:02d}.json"
                out.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  ✓ Lưu: {out.name} ({len(article['content'])} chars)")


if __name__ == "__main__":
    if not any(u for u in ARTICLE_URLS if u and not u.startswith("#")):
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Tìm bài báo về 'nghệ sĩ ma tuý' trên vnexpress.net, tuoitre.vn, dantri.com.vn")
    else:
        asyncio.run(crawl_all())
