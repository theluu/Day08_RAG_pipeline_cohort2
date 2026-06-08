# Task 2 — Crawl Bài Báo: Design Spec

**Date:** 2026-06-08  
**Status:** Approved

## Goal

Crawl tối thiểu 5 bài báo tiếng Việt về nghệ sĩ liên quan đến ma tuý. Lưu mỗi bài thành 1 file JSON vào `data/landing/news/`.

## Approach

Static URL list — pre-define 8 URL bài báo cụ thể, crawl bằng Crawl4AI's `AsyncWebCrawler`. Đơn giản, deterministic, không cần API key ngoài.

## Architecture

```
src/task2_crawl_news.py   ← script duy nhất
data/landing/news/        ← output directory
```

### Article URLs (8 bài)

Các vụ nghệ sĩ Việt liên quan ma tuý được đưa tin rộng rãi:
1. Châu Việt Cường — bị bắt 2018 (VnExpress)
2. Châu Việt Cường — phiên toà xét xử (Tuổi Trẻ)
3. Phúc XO — bị bắt (VnExpress hoặc Thanh Niên)
4. Tim (ca sĩ) — vụ việc 2021 (Kenh14 hoặc VnExpress)
5. Không Tú (rapper) — liên quan ma tuý
6. Tùng Dương / các nghệ sĩ khác — bài tổng hợp
7. Vụ bắt nhóm nghệ sĩ tại bar/club
8. Bài tổng hợp các nghệ sĩ Việt liên quan ma tuý

*URL cụ thể sẽ được xác định bằng WebSearch trước khi crawl.*

## Output JSON Schema

```json
{
  "url": "https://...",
  "title": "Tiêu đề bài báo",
  "crawl_date": "2026-06-08T11:00:00",
  "source_domain": "vnexpress.net",
  "content_markdown": "Nội dung bài báo dạng markdown...",
  "content_length": 3500
}
```

**Tên file:** `<slug-từ-title>.json` (lowercase, dấu gạch ngang, tối đa 60 ký tự)

## Implementation

```python
# src/task2_crawl_news.py
ARTICLE_URLS = [...]          # 8 URLs

async def crawl_article(url)  # → dict với schema trên
async def crawl_all(urls, output_dir)  # loop + save
def main()                    # asyncio.run(crawl_all(...))
```

- Dùng `AsyncWebCrawler` từ `crawl4ai`
- `result.markdown` → `content_markdown`
- `result.metadata.get("title")` → `title`
- Timeout 30s/bài, bỏ qua lỗi và tiếp tục
- In summary cuối: bao nhiêu bài thành công / thất bại

## Dependencies

```
crawl4ai>=0.3.0   # đã có trong requirements.txt
playwright        # cài tự động qua crawl4ai[playwright]
```

Cài: `pip install crawl4ai && crawl4ai-setup`

## Success Criteria

- Ít nhất 5 file JSON hợp lệ trong `data/landing/news/`
- Mỗi file có đủ 6 trường schema
- `content_length > 200` (nội dung thực chất, không phải trang lỗi)
