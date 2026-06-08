"""
Task 5 — Semantic Search Module (Dense Retrieval).

Query ChromaDB với vector embedding. Score = 1/(1+L2_distance) ∈ (0,1].
"""
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    collection = _client.get_or_create_collection(COLLECTION_NAME)
    count = collection.count()
    if count == 0:
        return []

    query_embedding = _model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content": doc,
                "score": 1.0 / (1.0 + dist),
                "metadata": meta or {},
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
