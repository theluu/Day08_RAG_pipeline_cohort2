"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex

PageIndex không dùng vector embeddings — dùng structural understanding
của document (tree hierarchy) để retrieve relevant passages.

Flow upload:
  submit_document(pdf_path) → doc_id  [async processing]
  is_retrieval_ready(doc_id) → bool   [poll until True]

Flow query (async 2 bước):
  submit_query(doc_id, query) → retrieval_id
  get_retrieval(retrieval_id) → results [poll until completed]

Tài liệu upload: data/landing/legal/*.pdf (PageIndex Cloud hiện nhận PDF)
Doc IDs được cache tại data/pageindex_doc_ids.json để tránh upload lại.
"""
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_DOC_IDS = os.getenv("PAGEINDEX_DOC_IDS", "")
PDF_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

POLL_INTERVAL = 3    # giây giữa mỗi lần poll
POLL_TIMEOUT = 120   # tối đa chờ 2 phút


def _get_client():
    from pageindex import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _parse_env_doc_ids(value: str) -> dict[str, str]:
    """
    Optional manual cache for already-uploaded PageIndex docs.

    Supports either JSON object:
      PAGEINDEX_DOC_IDS='{"file.pdf":"doc_123"}'
    or comma-separated doc IDs:
      PAGEINDEX_DOC_IDS='doc_123,doc_456'
    """
    if not value.strip():
        return {}

    try:
        parsed: Any = json.loads(value)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
        if isinstance(parsed, list):
            return {f"env_doc_{i + 1}": str(v) for i, v in enumerate(parsed)}
    except json.JSONDecodeError:
        pass

    return {
        f"env_doc_{i + 1}": doc_id.strip()
        for i, doc_id in enumerate(value.split(","))
        if doc_id.strip()
    }


def _load_doc_ids() -> dict[str, str]:
    """Load cached {filename → doc_id} từ JSON file."""
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return _parse_env_doc_ids(PAGEINDEX_DOC_IDS)


def _save_doc_ids(mapping: dict[str, str]) -> None:
    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))


def _extract_doc_id(response: dict) -> str:
    """Lấy doc_id từ response upload của PageIndex SDK."""
    doc_id = response.get("doc_id") or response.get("id")
    if not doc_id:
        raise RuntimeError(f"PageIndex upload response thiếu doc_id: {response}")
    return str(doc_id)


def _get_item_content(item) -> str:
    """Chuẩn hóa content từ retrieved node/string của PageIndex."""
    if not isinstance(item, dict):
        return str(item)

    for key in ("content", "text", "markdown", "page_content", "node_content"):
        value = item.get(key)
        if value:
            return str(value)
    return str(item)


def _get_item_score(item) -> float:
    """PageIndex retrieval có thể không trả score; default 1.0 cho fallback."""
    if not isinstance(item, dict):
        return 1.0

    for key in ("score", "relevance_score", "confidence"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _extract_retrieved_items(response: dict) -> list:
    """Lấy danh sách retrieved nodes từ response get_retrieval()."""
    for key in ("retrieved_nodes", "results", "data", "nodes"):
        value = response.get(key)
        if isinstance(value, list):
            return value
    return []


def upload_documents() -> dict[str, str]:
    """
    Upload PDF files lên PageIndex. Skip files đã upload (dùng cached doc_ids).

    Returns:
        {filename: doc_id} mapping
    """
    client = _get_client()
    cached = _load_doc_ids()

    for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
        if pdf_file.name in cached:
            print(f"  ↷ Already uploaded: {pdf_file.name} ({cached[pdf_file.name]})")
            continue
        try:
            result = client.submit_document(file_path=str(pdf_file))
            doc_id = _extract_doc_id(result)
            cached[pdf_file.name] = doc_id
            print(f"  ✓ Uploaded: {pdf_file.name} → {doc_id}")
        except Exception as exc:
            print(f"  ✗ Upload failed: {pdf_file.name}: {exc}")

    _save_doc_ids(cached)
    return cached


def _wait_ready(client, doc_id: str, timeout: int = POLL_TIMEOUT) -> bool:
    """Poll is_retrieval_ready() cho đến khi True hoặc timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.is_retrieval_ready(doc_id):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _query_one_doc(client, doc_id: str, query: str) -> list[dict]:
    """Submit query cho 1 doc và poll kết quả."""
    resp = client.submit_query(doc_id=doc_id, query=query)
    retrieval_id = resp.get("retrieval_id") or resp.get("id")
    if not retrieval_id:
        return []

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status == "completed":
            return _extract_retrieved_items(result)
        if status == "failed":
            print(f"  ✗ Retrieval failed for doc {doc_id}")
            return []
        time.sleep(POLL_INTERVAL)

    print(f"  ⚠ Timeout waiting for retrieval {retrieval_id}")
    return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả phù hợp.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả trả về

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
        sorted by score descending.
    """
    if not PAGEINDEX_API_KEY:
        return []

    cached = _load_doc_ids()
    if not cached:
        print("  Chưa có doc_ids. Chạy upload_documents() trước.")
        return []

    client = _get_client()
    all_results: list[dict] = []

    for filename, doc_id in cached.items():
        try:
            if not client.is_retrieval_ready(doc_id):
                print(f"  ⚠ {filename} chưa ready, skip")
                continue
            raw_items = _query_one_doc(client, doc_id, query)
        except Exception as exc:
            print(f"  PageIndex search error for {filename}: {exc}")
            continue

        for item in raw_items:
            metadata = {
                "source": filename,
                "doc_id": doc_id,
            }
            if isinstance(item, dict):
                metadata.update({
                    k: v for k, v in item.items()
                    if k not in (
                        "content", "text", "markdown", "page_content",
                        "node_content", "score", "relevance_score", "confidence",
                    )
                })
            all_results.append({
                "content": _get_item_content(item),
                "score": _get_item_score(item),
                "metadata": metadata,
                "source": "pageindex",
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("── Uploading documents ──")
        doc_ids = upload_documents()
        print(f"\nCached {len(doc_ids)} doc_ids:\n  {doc_ids}")

        print("\n── Waiting for documents to be ready ──")
        client = _get_client()
        for name, did in doc_ids.items():
            ready = _wait_ready(client, did, timeout=60)
            print(f"  {name}: {'✓ ready' if ready else '✗ timeout'}")

        print("\n── Test query ──")
        results = pageindex_search("hình phạt tội tàng trữ ma tuý", top_k=3)
        if results:
            for r in results:
                print(f"  [{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
        else:
            print("  Không có kết quả (kiểm tra API key và doc readiness)")
