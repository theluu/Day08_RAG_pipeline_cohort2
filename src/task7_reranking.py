"""
Task 7 — Reranking Module.

Hai thành phần:
  rerank_rrf: Reciprocal Rank Fusion — gộp nhiều ranked lists thành 1.
    score_rrf(d) = Σ_lists 1/(k + rank_i(d)),  k=60 (Cormack et al. 2009)
    Preserves best original score per doc for downstream threshold comparison.

  rerank: Cross-encoder via Jina Reranker API.
    Fallback: sort by original score nếu không có JINA_API_KEY.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_URL = "https://api.jina.ai/v1/rerank"


def rerank_rrf(results_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — merge multiple ranked lists into one.

    Args:
        results_lists: List of ranked result lists. Each item must have 'content'.
        k: RRF constant (60 is the standard default per Cormack et al.)

    Returns:
        Merged list sorted by RRF score descending.
        Each item keeps its best original 'score' alongside 'rrf_score'.
    """
    rrf_scores: dict[str, float] = {}
    best_orig: dict[str, float] = {}
    docs_map: dict[str, dict] = {}

    for ranked_list in results_lists:
        for rank, item in enumerate(ranked_list):
            key = item["content"][:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            orig = item.get("score", 0.0)
            best_orig[key] = max(best_orig.get(key, 0.0), orig)
            docs_map[key] = item

    merged = []
    for key, rrf in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(docs_map[key])
        item["score"] = best_orig[key]
        item["rrf_score"] = rrf
        merged.append(item)
    return merged


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates với Jina Reranker API (fallback: sort by score).

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số kết quả trả về

    Returns:
        List of top_k candidates re-sorted by relevance score descending.
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    try:
        resp = requests.post(
            JINA_URL,
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for r in resp.json()["results"]:
            item = dict(candidates[r["index"]])
            item["score"] = r["relevance_score"]
            out.append(item)
        return out
    except Exception as e:
        print(f"  Jina rerank failed ({e}), falling back to score sort")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search

    QUERY = "hình phạt tội tàng trữ ma túy"
    TOP_K = 5

    print(f"Query: {QUERY}\n")

    semantic = semantic_search(QUERY, top_k=TOP_K)
    lexical = lexical_search(QUERY, top_k=TOP_K)

    print("── Semantic (task5) ──")
    for r in semantic:
        print(f"  [{r['score']:.4f}] {r['content'][:90]}...")

    print("\n── BM25 (task6) ──")
    for r in lexical:
        print(f"  [{r['score']:.4f}] {r['content'][:90]}...")

    # RRF merge: gộp 2 ranked lists → 1 list theo Reciprocal Rank Fusion
    merged = rerank_rrf([semantic, lexical])
    print("\n── RRF merged (task7) ──")
    for r in merged[:TOP_K]:
        print(f"  [rrf={r['rrf_score']:.4f} orig={r['score']:.4f}] {r['content'][:90]}...")

    # Cross-encoder rerank (Jina API nếu có key, fallback sort by score)
    candidates = merged[:TOP_K * 2]
    reranked = rerank(QUERY, candidates, top_k=TOP_K)
    print("\n── Reranked (task7) ──")
    for r in reranked:
        print(f"  [{r['score']:.4f}] {r['content'][:90]}...")
