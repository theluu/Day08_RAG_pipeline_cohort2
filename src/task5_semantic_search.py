"""
Task 5 — Semantic Search Module (Dense Retrieval).

Dùng cùng embedding model với Task 4 (OpenAI text-embedding-3-small) để
encode query, rồi query ChromaDB collection đã index.

Score: cosine similarity = 1 - cosine_distance
- ChromaDB cosine space trả về distance ∈ [0, 2], trong đó:
  0 = vectors giống hệt, 1 = vuông góc, 2 = đối ngược
- score = 1 - distance → ∈ [-1, 1], cao hơn = liên quan hơn
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client: OpenAI | None = None
_collection = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY không tìm thấy trong .env")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed_query(query: str) -> list[float]:
    response = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng dense vector retrieval.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả trả về

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = _embed_query(query)
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
                "score": round(1.0 - dist, 4),
                "metadata": meta or {},
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "ca sĩ bị bắt vì liên quan ma túy",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = semantic_search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] ({r['metadata'].get('doc_type')}) {r['content'][:100]}...")
