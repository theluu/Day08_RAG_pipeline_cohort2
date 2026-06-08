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
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    ]
    for q in queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {result['sources']} | via {result['retrieval_source']}]")
