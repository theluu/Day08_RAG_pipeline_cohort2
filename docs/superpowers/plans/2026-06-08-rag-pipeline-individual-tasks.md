# RAG Pipeline v2 — Individual Tasks (1–10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all 10 individual tasks to build a complete end-to-end RAG pipeline on Vietnamese drug law documents and news articles, making `pytest tests/test_individual.py -v` fully pass.

**Architecture:** ChromaDB (local persistent, no Docker) for vector store, `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `rank-bm25` for lexical search, RRF merge + Jina reranker (with score-sort fallback), PageIndex for vectorless fallback, OpenAI gpt-4o-mini for generation with citation.

**Tech Stack:** `chromadb`, `sentence-transformers`, `rank-bm25`, `crawl4ai`, `markitdown`, `langchain-text-splitters`, `openai`, `python-dotenv`, `requests`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `requirements.txt` | Modify | Uncomment chromadb, comment weaviate |
| `.env` | Create | Copy from .env.example, fill API keys |
| `data/landing/legal/*.pdf` | Create (manual) | ≥3 raw legal PDFs |
| `data/landing/news/*.json` | Create (script) | ≥5 crawled news articles |
| `data/standardized/legal/*.md` | Create (script) | Converted legal docs |
| `data/standardized/news/*.md` | Create (script) | Converted news articles |
| `data/chroma_db/` | Create (script) | ChromaDB persistent storage |
| `src/task1_collect_legal_docs.py` | Modify | Download helper + instructions |
| `src/task2_crawl_news.py` | Modify | Crawl4AI async crawler |
| `src/task3_convert_markdown.py` | Modify | MarkItDown converter |
| `src/task4_chunking_indexing.py` | Modify | Chunking + ChromaDB indexing |
| `src/task5_semantic_search.py` | Modify | Dense retrieval from ChromaDB |
| `src/task6_lexical_search.py` | Modify | BM25 lexical search |
| `src/task7_reranking.py` | Modify | RRF merge + Jina reranking |
| `src/task8_pageindex_vectorless.py` | Modify | PageIndex API wrapper |
| `src/task9_retrieval_pipeline.py` | Modify | Full pipeline + fallback |
| `src/task10_generation.py` | Modify | LLM generation with citation |

---

## Task 0: Environment Setup

**Files:**
- Modify: `requirements.txt`
- Create: `.env`

- [ ] **Step 1: Update requirements.txt — swap Weaviate for ChromaDB**

In `requirements.txt`, change:
```
weaviate-client>=4.5.0
# chromadb>=0.4.0        # Alternative
```
to:
```
# weaviate-client>=4.5.0
chromadb>=0.4.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
crawl4ai-setup  # installs playwright browsers (required by crawl4ai)
```

Expected: all packages install without error.

- [ ] **Step 3: Create .env from template**

```bash
cp .env.example .env
```

Then open `.env` and fill in:
```
OPENAI_API_KEY=sk-...       # Required for Task 10
PAGEINDEX_API_KEY=pi_...    # Required for Task 8 (sign up at pageindex.ai)
JINA_API_KEY=jina_...       # Optional for Task 7 (fallback available)
```

---

## Task 1: Collect Legal Documents

**Files:**
- Modify: `src/task1_collect_legal_docs.py`
- Create (manual): `data/landing/legal/luat-phong-chong-ma-tuy-2021.pdf`
- Create (manual): `data/landing/legal/nghi-dinh-105-2021.pdf`
- Create (manual): `data/landing/legal/blhs-2015-chuong-xx.pdf`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask1 -v
```

Expected: 1 PASS (dir exists due to .gitkeep), 2 FAIL (no PDF files).

- [ ] **Step 2: Write task1_collect_legal_docs.py**

Replace the entire file content with:
```python
"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Nguồn tải thủ công:
    https://thuvienphapluat.vn — tìm theo số hiệu văn bản
"""
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Download file từ URL về DATA_DIR."""
    output_path = DATA_DIR / filename
    if output_path.exists():
        print(f"  Đã có: {filename}")
        return output_path
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  ✓ Đã tải: {filename} ({output_path.stat().st_size // 1024}KB)")
    return output_path


if __name__ == "__main__":
    setup_directory()
    print("\nHướng dẫn tải văn bản pháp luật:")
    print("1. Truy cập: https://thuvienphapluat.vn")
    print("2. Tìm và tải 3 văn bản sau về thư mục:", DATA_DIR)
    print()
    print("  a) 'Luật 73/2021/QH15' → luat-phong-chong-ma-tuy-2021.pdf")
    print("  b) 'Nghị định 105/2021/NĐ-CP' → nghi-dinh-105-2021.pdf")
    print("  c) 'Bộ luật Hình sự 2015 Chương XX' → blhs-2015-chuong-xx.pdf")
```

- [ ] **Step 3: Manually download 3 PDF files**

1. Go to https://thuvienphapluat.vn
2. Search "73/2021/QH15" → download PDF → save as `data/landing/legal/luat-phong-chong-ma-tuy-2021.pdf`
3. Search "105/2021/NĐ-CP" → download PDF → save as `data/landing/legal/nghi-dinh-105-2021.pdf`
4. Search "Bộ luật Hình sự 2015" → download the criminal code PDF → save as `data/landing/legal/blhs-2015-chuong-xx.pdf`

Each file must be > 1KB (real PDFs are typically 500KB–5MB).

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask1 -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/task1_collect_legal_docs.py data/landing/legal/
git commit -m "feat: task1 legal document collection (3 PDFs)"
```

---

## Task 2: Crawl News Articles

**Files:**
- Modify: `src/task2_crawl_news.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask2 -v
```

Expected: 1 PASS (dir exists), 3 FAIL (no news files).

- [ ] **Step 2: Write task2_crawl_news.py**

Replace the entire file content with:
```python
"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Điền URL thực tế của 5 bài báo — tìm trên vnexpress.net, tuoitre.vn, dantri.com.vn
# Tìm các bài báo về: nghệ sĩ bị bắt ma tuý, vụ án liên quan nghệ sĩ và ma tuý
ARTICLE_URLS = [
    "REPLACE_URL_1",  # vnexpress.net/...
    "REPLACE_URL_2",  # vnexpress.net/...
    "REPLACE_URL_3",  # tuoitre.vn/...
    "REPLACE_URL_4",  # dantri.com.vn/...
    "REPLACE_URL_5",  # dantri.com.vn/...
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(crawler: AsyncWebCrawler, url: str) -> dict | None:
    try:
        result = await crawler.arun(url=url)
        if not result.success:
            print(f"  ✗ Thất bại: {url}")
            return None
        return {
            "url": url,
            "title": result.metadata.get("title", ""),
            "date_crawled": datetime.now().isoformat(),
            "content": result.markdown or result.cleaned_html or "",
        }
    except Exception as e:
        print(f"  ✗ Lỗi {url}: {e}")
        return None


async def crawl_all():
    setup_directory()
    async with AsyncWebCrawler(verbose=False) as crawler:
        for i, url in enumerate(ARTICLE_URLS):
            if url.startswith("REPLACE"):
                print(f"  ⚠ Chưa điền URL {i+1}")
                continue
            print(f"[{i+1}/{len(ARTICLE_URLS)}] {url}")
            article = await crawl_article(crawler, url)
            if article:
                out = DATA_DIR / f"article_{i+1:02d}.json"
                out.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"  ✓ Lưu: {out.name} ({len(article['content'])} chars)")


if __name__ == "__main__":
    asyncio.run(crawl_all())
```

- [ ] **Step 3: Fill in real article URLs and run**

1. Search on vnexpress.net, tuoitre.vn, dantri.com.vn for news about Vietnamese artists involved in drug cases
2. Replace `REPLACE_URL_1` through `REPLACE_URL_5` with 5 real article URLs
3. Run the crawler:

```bash
python src/task2_crawl_news.py
```

Expected: 5 files created in `data/landing/news/`, each > 500 bytes.

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask2 -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/task2_crawl_news.py data/landing/news/
git commit -m "feat: task2 crawl 5 news articles about artists and drugs"
```

---

## Task 3: Convert to Markdown

**Files:**
- Modify: `src/task3_convert_markdown.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask3 -v
```

Expected: 1 PASS (dir exists), 3 FAIL (no markdown files).

- [ ] **Step 2: Write task3_convert_markdown.py**

Replace the entire file content with:
```python
"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown (Microsoft): pip install markitdown
"""
import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

_md = MarkItDown()


def convert_legal_docs():
    """Convert PDF/DOCX trong data/landing/legal/ sang .md."""
    legal_dir = LANDING_DIR / "legal"
    out_dir = OUTPUT_DIR / "legal"
    out_dir.mkdir(parents=True, exist_ok=True)

    supported = {".pdf", ".docx", ".doc"}
    converted = 0
    for src in legal_dir.iterdir():
        if src.suffix.lower() not in supported:
            continue
        out = out_dir / (src.stem + ".md")
        print(f"  Converting: {src.name}")
        result = _md.convert(str(src))
        out.write_text(result.text_content, encoding="utf-8")
        print(f"  ✓ {out.name} ({len(result.text_content)} chars)")
        converted += 1
    return converted


def convert_news_articles():
    """Convert JSON news files sang .md với metadata header."""
    news_dir = LANDING_DIR / "news"
    out_dir = OUTPUT_DIR / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for src in news_dir.iterdir():
        if src.suffix != ".json":
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        out = out_dir / (src.stem + ".md")
        lines = [
            f"# {data.get('title', src.stem)}",
            f"",
            f"**URL:** {data.get('url', '')}",
            f"**Crawled:** {data.get('date_crawled', '')}",
            f"",
            data.get("content", ""),
        ]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ {out.name}")
        converted += 1
    return converted


if __name__ == "__main__":
    print("Converting legal docs...")
    n1 = convert_legal_docs()
    print(f"\nConverting news articles...")
    n2 = convert_news_articles()
    print(f"\nDone: {n1} legal + {n2} news = {n1+n2} total files")
```

- [ ] **Step 3: Run conversion**

```bash
python src/task3_convert_markdown.py
```

Expected output like:
```
Converting legal docs...
  Converting: luat-phong-chong-ma-tuy-2021.pdf
  ✓ luat-phong-chong-ma-tuy-2021.md (45231 chars)
  ...
Converting news articles...
  ✓ article_01.md
  ...
Done: 3 legal + 5 news = 8 total files
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask3 -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/task3_convert_markdown.py data/standardized/
git commit -m "feat: task3 convert all documents to markdown"
```

---

## Task 4: Chunking & Indexing

**Files:**
- Modify: `src/task4_chunking_indexing.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask4 -v
```

Expected: SKIP (NotImplementedError).

- [ ] **Step 2: Replace task4_chunking_indexing.py**

Replace the entire file:
```python
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
```

- [ ] **Step 3: Run indexing**

```bash
python src/task4_chunking_indexing.py
```

Expected:
```
Loading documents...
  8 docs
  N chunks
Loading model: sentence-transformers/all-MiniLM-L6-v2
✓ Indexed N chunks into ChromaDB at .../data/chroma_db
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask4 -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/task4_chunking_indexing.py
git commit -m "feat: task4 chunking and ChromaDB indexing with all-MiniLM-L6-v2"
```

---

## Task 5: Semantic Search

**Files:**
- Modify: `src/task5_semantic_search.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask5 -v
```

Expected: SKIP (no implementation).

- [ ] **Step 2: Replace task5_semantic_search.py**

```python
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

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    collection = _client.get_or_create_collection(COLLECTION_NAME)
    count = collection.count()
    if count == 0:
        return []

    query_embedding = _model.encode(query).tolist()
    n = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
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
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask5 -v
```

Expected: 4 tests PASS (requires task4 indexing to have run).

- [ ] **Step 4: Commit**

```bash
git add src/task5_semantic_search.py
git commit -m "feat: task5 semantic search with ChromaDB"
```

---

## Task 6: Lexical Search (BM25)

**Files:**
- Modify: `src/task6_lexical_search.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask6 -v
```

Expected: SKIP.

- [ ] **Step 2: Replace task6_lexical_search.py**

```python
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
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask6 -v
```

Expected: 4 tests PASS (requires data in `data/standardized/`).

- [ ] **Step 4: Commit**

```bash
git add src/task6_lexical_search.py
git commit -m "feat: task6 BM25 lexical search"
```

---

## Task 7: Reranking

**Files:**
- Modify: `src/task7_reranking.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask7 -v
```

Expected: SKIP.

- [ ] **Step 2: Replace task7_reranking.py**

```python
"""
Task 7 — Reranking Module.

Hai hàm:
  rerank_rrf: Reciprocal Rank Fusion — gộp nhiều ranked lists
    score_rrf(d) = Σ_lists 1/(k + rank_i(d)),  k=60 (Cormack et al. 2009)
    Preserves the best original score for threshold comparison downstream.

  rerank: Cross-encoder via Jina Reranker API
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
        k: RRF constant (60 is standard default).

    Returns:
        Merged list sorted by RRF score descending.
        Each item preserves original 'score' (best across lists) alongside 'rrf_score'.
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
        item["score"] = best_orig[key]   # keep best original score for threshold checks
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
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask7 -v
```

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/task7_reranking.py
git commit -m "feat: task7 RRF merge and Jina reranking with fallback"
```

---

## Task 8: PageIndex Vectorless RAG

**Files:**
- Modify: `src/task8_pageindex_vectorless.py`

- [ ] **Step 1: Sign up for PageIndex**

1. Go to https://pageindex.ai/ and create an account
2. Get API key from dashboard
3. Add to `.env`: `PAGEINDEX_API_KEY=pi_...`

- [ ] **Step 2: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask8 -v
```

Expected: 1 PASS (function exists if imported), 1 SKIP/FAIL.

- [ ] **Step 3: Replace task8_pageindex_vectorless.py**

```python
"""
Task 8 — PageIndex Vectorless RAG.

PageIndex không dùng vector embeddings — thay vào đó dùng structural
understanding của document để trả lời câu hỏi. Dùng làm fallback khi
hybrid search cho kết quả kém.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_client():
    import pageindex
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được set trong .env")
    return pageindex.Client(api_key=PAGEINDEX_API_KEY)


def upload_documents() -> list[str]:
    """Upload standardized markdown files to PageIndex. Returns list of doc IDs."""
    client = _get_client()
    doc_ids = []
    for doc_type in ("legal", "news"):
        subdir = STANDARDIZED_DIR / doc_type
        if not subdir.exists():
            continue
        for md_file in subdir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            result = client.upload(content=content, filename=md_file.name)
            doc_ids.append(result.id)
            print(f"  ✓ Uploaded: {md_file.name}")
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả đủ tốt.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY:
        return []
    try:
        client = _get_client()
        results = client.query(query=query, top_k=top_k)
        output = []
        for r in results:
            output.append({
                "content": getattr(r, "content", str(r)),
                "score": float(getattr(r, "score", 1.0)),
                "metadata": {"filename": getattr(r, "filename", "")},
                "source": "pageindex",
            })
        return output
    except Exception as e:
        print(f"  PageIndex search error: {e}")
        return []


if __name__ == "__main__":
    print("Uploading to PageIndex...")
    ids = upload_documents()
    print(f"Uploaded {len(ids)} documents")
    print("\nTest search:")
    results = pageindex_search("hình phạt ma tuý", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:80]}")
```

- [ ] **Step 4: Upload documents (requires API key)**

```bash
python src/task8_pageindex_vectorless.py
```

- [ ] **Step 5: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask8 -v
```

Expected: 2 tests PASS (or 1 PASS + 1 SKIP if no API key — acceptable).

- [ ] **Step 6: Commit**

```bash
git add src/task8_pageindex_vectorless.py
git commit -m "feat: task8 PageIndex vectorless RAG with fallback"
```

---

## Task 9: Full Retrieval Pipeline

**Files:**
- Modify: `src/task9_retrieval_pipeline.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask9 -v
```

Expected: SKIP (NotImplementedError).

- [ ] **Step 2: Replace task9_retrieval_pipeline.py**

```python
"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline logic:
  1. semantic_search + lexical_search (song song theo luồng code)
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

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source is 'hybrid' or 'pageindex'.
    """
    semantic_results = semantic_search(query, top_k=top_k * 2)
    lexical_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([semantic_results, lexical_results]) if (
        semantic_results or lexical_results
    ) else []

    candidates = merged[:top_k * 3]
    reranked = rerank(query, candidates, top_k=top_k) if candidates else []

    # Tag hybrid source before threshold check
    for r in reranked:
        r["source"] = "hybrid"

    # Use best original semantic score for threshold check
    best_score = reranked[0]["score"] if reranked else 0.0

    if best_score < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for r in fallback:
                r["source"] = "pageindex"
            return fallback[:top_k]

    return reranked[:top_k]
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask9 -v
```

Expected: 4 tests PASS.

Note: `test_results_have_required_keys` may SKIP if vector store has no data — ensure task4 has been run first.

- [ ] **Step 4: Commit**

```bash
git add src/task9_retrieval_pipeline.py
git commit -m "feat: task9 full retrieval pipeline with hybrid search and PageIndex fallback"
```

---

## Task 10: Generation with Citation

**Files:**
- Modify: `src/task10_generation.py`

- [ ] **Step 1: Run tests to see current state**

```bash
pytest tests/test_individual.py::TestTask10 -v
```

Expected: SKIP (NotImplementedError).

- [ ] **Step 2: Replace task10_generation.py**

```python
"""
Task 10 — Generation Có Citation.

top_k=5: đủ evidence mà không gây lost in the middle (5 chunks ≈ 2500 chars context)
top_p=0.9: nucleus sampling — diverse nhưng không quá ngẫu nhiên cho factual QA
temperature=0.3: thấp để accurate khi cite facts, tránh hallucination
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from .task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source
(e.g., [Author/Platform Name, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" (Liu et al. 2023):
    chunk quan trọng nhất ở đầu và cuối, ít quan trọng hơn ở giữa.

    Ví dụ với 5 chunks [0,1,2,3,4]: output [0,2,4,3,1]
    - Even index i → đặt từ đầu vào (positions 0,1,2...)
    - Odd index i → đặt từ cuối vào (...,n-2,n-1)
    """
    if len(chunks) <= 2:
        return list(chunks)

    n = len(chunks)
    result: list = [None] * n
    left, right = 0, n - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return result


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string với source label cho citation.

    Mỗi chunk được đánh dấu [Source N: filename] để LLM có thể cite.
    """
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("metadata", {}).get("source", "unknown")
        source_name = source.replace(".md", "").replace(".pdf", "")
        parts.append(f"[Source {i+1}: {source_name}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
) -> dict:
    """
    Generate answer có citation từ context chunks.

    Args:
        query: Câu hỏi
        context_chunks: Optional pre-retrieved chunks. If None, calls retrieve().

    Returns:
        {'answer': str, 'sources': list[str], 'retrieval_source': str}
    """
    if context_chunks is None:
        context_chunks = retrieve(query, top_k=TOP_K)

    if not context_chunks:
        return {
            "answer": "I cannot verify this information",
            "sources": [],
            "retrieval_source": "none",
        }

    ordered = reorder_for_llm(context_chunks)
    context = format_context(ordered)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content
    sources = list({c.get("metadata", {}).get("source", "unknown") for c in context_chunks})
    retrieval_source = context_chunks[0].get("source", "hybrid") if context_chunks else "none"

    return {"answer": answer, "sources": sources, "retrieval_source": retrieval_source}


if __name__ == "__main__":
    q = "Hình phạt tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?"
    result = generate_with_citation(q)
    print(result["answer"])
    print(f"\n[Sources: {result['sources']}]")
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/test_individual.py::TestTask10 -v
```

Expected: 3 tests PASS (`test_generate_returns_dict_with_answer` may SKIP without OpenAI key — acceptable).

- [ ] **Step 4: Commit**

```bash
git add src/task10_generation.py
git commit -m "feat: task10 generation with citation and lost-in-middle prevention"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
pytest tests/test_individual.py -v
```

Expected minimum pass count: 30+ tests PASS, ≤ 3 SKIP (API-dependent).

Score breakdown:
- Task 1: 3/3 pts ✓
- Task 2: 3/3 pts ✓
- Task 3: 4/4 pts ✓
- Task 4: 7/7 pts ✓
- Task 5: 6/6 pts ✓
- Task 6: 6/6 pts ✓
- Task 7: 6/6 pts ✓
- Task 8: 4/4 pts (2/4 if no PageIndex key)
- Task 9: 7/7 pts ✓
- Task 10: 4/4 pts (3/4 if no OpenAI key)

**Total: 46–50/50 pts depending on API keys.**

- [ ] **Commit final state**

```bash
git add -A
git commit -m "chore: complete all 10 individual tasks"
```
