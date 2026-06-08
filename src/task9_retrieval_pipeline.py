"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline logic:
  1. Chạy semantic_search (Task 5) và lexical_search (Task 6)
  2. Merge bằng Reciprocal Rank Fusion (Task 7)
  3. Rerank bằng Jina reranker hoặc fallback score-sort (Task 7)
  4. Nếu top result score < threshold thì fallback PageIndex (Task 8)
  5. Return top_k results với source tag: "hybrid" hoặc "pageindex"
"""
from __future__ import annotations

from collections.abc import Callable

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def _safe_call(name: str, fn: Callable[[], list[dict]]) -> list[dict]:
    """Run one retrieval step without letting a failing module break fallback."""
    try:
        results = fn()
    except Exception as exc:
        print(f"  {name} failed: {exc}")
        return []
    return results if isinstance(results, list) else []


def _semantic_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task5_semantic_search import semantic_search
    except ImportError:
        from task5_semantic_search import semantic_search
    return semantic_search(query, top_k=top_k)


def _lexical_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task6_lexical_search import lexical_search
    except ImportError:
        from task6_lexical_search import lexical_search
    return lexical_search(query, top_k=top_k)


def _merge_results(results_lists: list[list[dict]]) -> list[dict]:
    try:
        from .task7_reranking import rerank_rrf
    except ImportError:
        from task7_reranking import rerank_rrf
    return rerank_rrf([results for results in results_lists if results])


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    try:
        from .task7_reranking import rerank
    except ImportError:
        from task7_reranking import rerank
    return rerank(query, candidates, top_k=top_k)


def _pageindex_fallback(query: str, top_k: int) -> list[dict]:
    try:
        from .task8_pageindex_vectorless import pageindex_search
    except ImportError:
        from task8_pageindex_vectorless import pageindex_search
    return pageindex_search(query, top_k=top_k)


def _tag_source(results: list[dict], source: str) -> list[dict]:
    tagged = []
    for item in results:
        result = dict(item)
        result["source"] = source
        tagged.append(result)
    return tagged


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """
    Full retrieval: hybrid search → merge → rerank → fallback PageIndex.

    1. Chạy semantic_search + lexical_search
    2. Merge kết quả bằng RRF
    3. Rerank
    4. Nếu top result score < threshold → fallback PageIndex
    5. Return top_k results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Nếu best score < threshold thì fallback sang PageIndex

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source là "hybrid" hoặc "pageindex".
    """
    if top_k <= 0 or not query.strip():
        return []

    search_k = top_k * 2
    semantic_results = _safe_call(
        "semantic_search",
        lambda: _semantic_search(query, top_k=search_k),
    )
    lexical_results = _safe_call(
        "lexical_search",
        lambda: _lexical_search(query, top_k=search_k),
    )

    merged = []
    if semantic_results or lexical_results:
        merged = _safe_call(
            "rrf_merge",
            lambda: _merge_results([semantic_results, lexical_results]),
        )

    candidates = merged[:top_k * 3]
    reranked = _safe_call(
        "rerank",
        lambda: _rerank(query, candidates, top_k=top_k),
    ) if candidates else []

    hybrid_results = _tag_source(reranked[:top_k], "hybrid")
    best_score = float(hybrid_results[0].get("score", 0.0)) if hybrid_results else 0.0

    if best_score < score_threshold:
        fallback = _safe_call(
            "pageindex_search",
            lambda: _pageindex_fallback(query, top_k=top_k),
        )
        if fallback:
            return _tag_source(fallback[:top_k], "pageindex")

    return hybrid_results


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
