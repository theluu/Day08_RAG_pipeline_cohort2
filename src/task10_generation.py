"""
Task 10 — Generation Có Citation.

TOP_K=5: lấy 5 chunks sau reranking để có đủ evidence cho câu hỏi pháp lý/tin tức,
nhưng vẫn giữ context ngắn để giảm "lost in the middle".
TOP_P=0.2: factual QA cần câu trả lời ổn định, ít suy diễn; nucleus sampling thấp
giúp mô hình tập trung vào các token có xác suất cao trong context.
TEMPERATURE=0.2: giảm hallucination khi bắt buộc cite từng claim.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # Allows running directly: python src/task10_generation.py
    from task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.2
TEMPERATURE = 0.2
MODEL = "gpt-4o-mini"
UNVERIFIABLE_ANSWER = "I cannot verify this information"

SYSTEM_PROMPT = """Answer the following question comprehensively.
Use only the provided context.
For every statement of fact or claim, immediately insert a citation
in brackets using the exact citation labels shown in the context
(for example: [VnExpress, 2026] or [Luật Phòng Chống Ma Túy, 2021]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle":
    chunk quan trọng nhất ở đầu và cuối, ít quan trọng hơn ở giữa.

    Ví dụ với 5 chunks [1,2,3,4,5] → [1,3,5,4,2].
    """
    if len(chunks) <= 2:
        return list(chunks)

    result: list[dict | None] = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            result[left] = chunk
            left += 1
        else:
            result[right] = chunk
            right -= 1
    return [chunk for chunk in result if chunk is not None]


def _source_name(source: str) -> str:
    stem = Path(source).stem if source else "unknown"
    cleaned = re.sub(r"[-_]+", " ", stem).strip()
    return cleaned.title() if cleaned else "Unknown"


def _source_year(chunk: dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    candidates = [
        str(metadata.get("year", "")),
        str(metadata.get("date", "")),
        str(metadata.get("crawled", "")),
        str(metadata.get("source", "")),
        chunk.get("content", "")[:1200],
    ]
    years = []
    for text in candidates:
        years.extend(int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", text))
    return str(max(years)) if years else "Năm không rõ"


def citation_label(chunk: dict) -> str:
    """Return citation label dạng [Nguồn, Năm] để LLM dùng nguyên văn."""
    source = chunk.get("metadata", {}).get("source", "unknown")
    return f"[{_source_name(source)}, {_source_year(chunk)}]"


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string có citation label [Nguồn, Năm].

    Mỗi chunk có một label rõ ràng để LLM cite đúng nguồn thay vì tự bịa citation.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}
        source = metadata.get("source", "unknown")
        label = citation_label(chunk)
        content = chunk.get("content", "").strip()
        if not content:
            continue
        parts.append(
            f"[Context {i}]\n"
            f"Citation label: {label}\n"
            f"Source file: {source}\n"
            f"Retrieval source: {chunk.get('source', 'hybrid')}\n"
            f"Content:\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def _has_evidence(context_chunks: list[dict]) -> bool:
    return any(chunk.get("content", "").strip() for chunk in context_chunks)


def _needs_citation(answer: str) -> bool:
    if answer.strip() == UNVERIFIABLE_ANSWER:
        return False
    return not bool(re.search(r"\[[^\[\],]+,\s*(?:19|20)\d{2}\]", answer))


def _call_llm(query: str, context: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return UNVERIFIABLE_ANSWER

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {query}\n\n"
                        "Answer in Vietnamese when the question is Vietnamese. "
                        "Every factual sentence must include one of the citation labels above."
                    ),
                },
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=1000,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  OpenAI generation failed: {exc}")
        return UNVERIFIABLE_ANSWER


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
) -> dict:
    """
    Generate answer có citation từ context chunks.

    1. Reorder chunks để tránh lost in the middle
    2. Format context với source metadata và citation label [Nguồn, Năm]
    3. Inject vào prompt với SYSTEM_PROMPT
    4. Gọi OpenAI LLM
    5. Return answer có citation, hoặc "I cannot verify this information"
    """
    if context_chunks is None:
        context_chunks = retrieve(query, top_k=TOP_K)

    if not context_chunks or not _has_evidence(context_chunks):
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
        }

    ordered = reorder_for_llm(context_chunks[:TOP_K])
    context = format_context(ordered)
    if not context:
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
        }

    answer = _call_llm(query, context)
    if _needs_citation(answer):
        answer = UNVERIFIABLE_ANSWER

    sources = sorted({
        chunk.get("metadata", {}).get("source", "unknown")
        for chunk in context_chunks
    })
    retrieval_source = context_chunks[0].get("source", "hybrid")

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    ]
    for q in queries:
        print(f"\n{'=' * 70}\nQ: {q}\n{'=' * 70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {result['sources']} | via {result['retrieval_source']}]")
