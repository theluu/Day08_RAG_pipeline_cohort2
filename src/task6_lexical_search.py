"""
Task 6 — Lexical Search Module (BM25).

BM25 (Okapi BM25) — Robertson & Sparck Jones:
  score(q,d) = Σ IDF(qi) × (tf(qi,d)×(k1+1)) / (tf(qi,d) + k1×(1-b+b×|d|/avgdl))
  k1=1.5 (term saturation), b=0.75 (length normalization)

Tokenize: lowercase + whitespace split — đủ tốt cho tiếng Việt domain pháp lý.
BM25 index được build lúc import module, load từ data/standardized/.
"""
from pathlib import Path

from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _load_corpus() -> list[dict]:
    corpus = []
    for doc_type in ("legal", "news"):
        subdir = STANDARDIZED_DIR / doc_type
        if not subdir.exists():
            continue
        for md_file in subdir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                corpus.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "doc_type": doc_type},
                })
    return corpus


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


_corpus = _load_corpus()
_tokenized = [_tokenize(d["content"]) for d in _corpus]
_bm25 = BM25Okapi(_tokenized) if _tokenized else None


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 lexical search.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by BM25 score descending.
    """
    if _bm25 is None or not _corpus:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(
        range(len(_corpus)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "content": _corpus[i]["content"],
            "score": float(scores[i]),
            "metadata": _corpus[i]["metadata"],
        }
        for i in ranked
    ]


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
