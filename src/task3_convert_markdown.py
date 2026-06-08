"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown (Microsoft): pip install markitdown
"""
import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

_md = MarkItDown()


def convert_legal_docs() -> int:
    """Convert PDF/DOCX trong data/landing/legal/ sang .md."""
    legal_dir = LANDING_DIR / "legal"
    out_dir = OUTPUT_DIR / "legal"
    out_dir.mkdir(parents=True, exist_ok=True)

    supported = {".pdf", ".docx", ".doc"}
    converted = 0
    for src in legal_dir.iterdir():
        if src.suffix.lower() not in supported:
            continue
        out = out_dir / (src.stem + ".md")
        print(f"  Converting: {src.name}")
        result = _md.convert(str(src))
        out.write_text(result.text_content, encoding="utf-8")
        print(f"  ✓ {out.name} ({len(result.text_content)} chars)")
        converted += 1
    return converted


def convert_news_articles() -> int:
    """Convert JSON news files sang .md với metadata header."""
    news_dir = LANDING_DIR / "news"
    out_dir = OUTPUT_DIR / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for src in news_dir.iterdir():
        if src.suffix != ".json":
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        out = out_dir / (src.stem + ".md")
        lines = [
            f"# {data.get('title', src.stem)}",
            "",
            f"**URL:** {data.get('url', '')}",
            f"**Crawled:** {data.get('crawl_date', data.get('date_crawled', ''))}",
            f"**Source:** {data.get('source_domain', '')}",
            "",
            data.get("content_markdown", data.get("content", "")),
        ]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ {out.name}")
        converted += 1
    return converted


def convert_all():
    print("Converting legal docs...")
    n1 = convert_legal_docs()
    print(f"\nConverting news articles...")
    n2 = convert_news_articles()
    print(f"\nDone: {n1} legal + {n2} news = {n1 + n2} total files")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
