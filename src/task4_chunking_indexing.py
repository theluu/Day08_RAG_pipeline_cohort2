"""
Task 4 — Chunking & Indexing vào ChromaDB.

Chunking: RecursiveCharacterTextSplitter
- chunk_size=800: đủ ngữ cảnh cho văn bản pháp lý/tin tức dài; nhỏ hơn 1000
  để tránh embedding mờ nghĩa khi đoạn văn quá dài
- chunk_overlap=100: ~12.5% overlap để giữ ngữ cảnh tại ranh giới chunk
- separators: ưu tiên tách ở \n\n (đoạn), \n (dòng), ". " (câu), rồi space

Embedding: OpenAI text-embedding-3-small
- 1536 dimensions, đa ngôn ngữ (kể cả tiếng Việt), gọi qua API
- Dùng API thay vì download model local để tiết kiệm thời gian cài đặt
- Cần OPENAI_API_KEY trong file .env

Vector Store: ChromaDB (local persistent)
- Không cần Docker, lưu vào data/chroma_db/
- Đủ cho single-node RAG; upgrade lên Weaviate khi cần hybrid search
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_STORE = "chromadb"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ", ", " ", ""],
)


def load_documents() -> list[dict]:
    """
    Đọc tất cả .md files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'doc_type': str}}
    """
    docs = []
    for doc_type in ("legal", "news"):
        subdir = STANDARDIZED_DIR / doc_type
        if not subdir.exists():
            continue
        for md_file in subdir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                docs.append({
                    "content": content,
                    "metadata": {"source": md_file.name, "doc_type": doc_type},
                })
    return docs


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chia documents thành chunks bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks = []
    for doc in documents:
        splits = _splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            chunks.append({
                "content": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Gọi OpenAI Embeddings API cho một batch texts."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_collection():
    """Get or create ChromaDB persistent collection."""
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents():
    """Load → chunk → embed (OpenAI) → upsert into ChromaDB."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY không tìm thấy. Tạo file .env với OPENAI_API_KEY=sk-...")

    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("⚠ Không có documents. Chạy task3 trước.")
        return

    print(f"  {len(docs)} docs")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    client = OpenAI(api_key=api_key)
    collection = get_collection()

    # OpenAI rate limit: batch tối đa 2048 inputs, nhưng giữ nhỏ để an toàn
    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = embed_texts(client, texts)
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"✓ Indexed {total} chunks into ChromaDB at {CHROMA_DIR}")
    print(f"  Model: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dims)")


if __name__ == "__main__":
    index_documents()
