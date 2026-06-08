"""
Task 4 — Chunking & Indexing vào ChromaDB.

Chunking: RecursiveCharacterTextSplitter
- chunk_size=500: đủ ngữ cảnh per chunk, không quá dài làm embedding mờ nghĩa
- chunk_overlap=50: giữ liên kết ngữ nghĩa tại ranh giới chunk (10% overlap)

Embedding: sentence-transformers/all-MiniLM-L6-v2
- 384 dimensions, chạy được CPU, không cần API key
- Đủ chất lượng cho retrieval tiếng Việt ở domain pháp lý cụ thể

Vector Store: ChromaDB (local persistent)
- Không cần Docker, lưu vào data/chroma_db/
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
VECTOR_STORE = "chromadb"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", ". ", " ", ""],
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


def get_collection():
    """Get or create ChromaDB persistent collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def index_documents():
    """Load → chunk → embed → upsert into ChromaDB."""
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("⚠ Không có documents. Chạy task3 trước.")
        return

    print(f"  {len(docs)} docs")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks")

    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    collection = get_collection()
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
    print(f"✓ Indexed {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    index_documents()
