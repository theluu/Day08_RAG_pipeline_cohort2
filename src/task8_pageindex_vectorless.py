"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex
Cài đặt: pip install pageindex

PageIndex không dùng vector embeddings — dùng structural understanding
của document để trả lời. Dùng làm fallback khi hybrid search kém.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_client():
    import pageindex
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set trong .env")
    return pageindex.Client(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> list[str]:
    """Upload standardized markdown files to PageIndex. Returns list of doc IDs."""
    client = _get_client()
    doc_ids = []
    for doc_type in ("legal", "news"):
        subdir = STANDARDIZED_DIR / doc_type
        if not subdir.exists():
            continue
        for md_file in subdir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            result = client.upload(content=content, filename=md_file.name)
            doc_ids.append(result.id)
            print(f"  ✓ Uploaded: {md_file.name}")
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả đủ tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY:
        return []
    try:
        client = _get_client()
        results = client.query(query=query, top_k=top_k)
        output = []
        for r in results:
            output.append({
                "content": getattr(r, "content", str(r)),
                "score": float(getattr(r, "score", 1.0)),
                "metadata": {"filename": getattr(r, "filename", "")},
                "source": "pageindex",
            })
        return output
    except Exception as e:
        print(f"  PageIndex search error: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        ids = upload_documents()
        print(f"Uploaded {len(ids)} documents")
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
