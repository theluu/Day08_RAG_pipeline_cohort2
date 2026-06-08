"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline logic:
  1. semantic_search + lexical_search
  2. RRF merge — preserves best original score per doc
  3. rerank (Jina API hoặc score-sort fallback)
  4. Nếu best score < score_threshold → fallback sang PageIndex
  5. Return với source tag: 'hybrid' hoặc 'pageindex'
"""
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """
    Full retrieval: hybrid search → rerank → fallback PageIndex.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Nếu best score < threshold → fallback sang PageIndex

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source là 'hybrid' hoặc 'pageindex'.
    """
    semantic_results = semantic_search(query, top_k=top_k * 2)
    lexical_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([semantic_results, lexical_results]) if (
        semantic_results or lexical_results
    ) else []

    candidates = merged[:top_k * 3]
    reranked = rerank(query, candidates, top_k=top_k) if candidates else []

    for r in reranked:
        r["source"] = "hybrid"

    best_score = reranked[0]["score"] if reranked else 0.0

    if best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for r in fallback:
                r["source"] = "pageindex"
            return fallback[:top_k]

    return reranked[:top_k]


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
