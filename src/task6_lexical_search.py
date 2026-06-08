"""
Task 6 — Lexical Search Module (BM25).

BM25 (Okapi BM25) — Robertson & Sparck Jones:
  score(q,d) = Σ IDF(qi) × tf(qi,d)×(k1+1) / (tf(qi,d) + k1×(1-b+b×|d|/avgdl))
  k1=1.5 (term saturation), b=0.75 (length normalization)

BM25 index build trên chunks (cùng đơn vị với task5/ChromaDB) để có thể
merge kết quả trong hybrid search ở task9.

Tokenize: lowercase + split() — đủ cho tiếng Việt đã được unicode normalize.
Index được build một lần lúc import module (lazy, chỉ khi hàm được gọi lần đầu).
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

try:
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:  # Allows running this file directly: python src/task6_lexical_search.py
    from task4_chunking_indexing import chunk_documents, load_documents

_chunks: list[dict] | None = None
_bm25: BM25Okapi | None = None


def _ensure_index() -> None:
    global _chunks, _bm25
    if _bm25 is not None:
        return
    _chunks = chunk_documents(load_documents())
    tokenized = [_tokenize(c["content"]) for c in _chunks]
    _bm25 = BM25Okapi(tokenized) if tokenized else None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 lexical search trên chunk corpus.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả trả về

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by BM25 score descending.
    """
    _ensure_index()
    if _bm25 is None or not _chunks:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(_chunks)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "content": _chunks[i]["content"],
            "score": round(float(scores[i]), 4),
            "metadata": _chunks[i]["metadata"],
        }
        for i in ranked_indices
        if scores[i] > 0
    ]


if __name__ == "__main__":
    queries = [
        "Điều 249 tàng trữ trái phép chất ma túy",
        "ca sĩ bị bắt liên quan ma túy",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = lexical_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] ({r['metadata'].get('doc_type')}) {r['content'][:100]}...")
