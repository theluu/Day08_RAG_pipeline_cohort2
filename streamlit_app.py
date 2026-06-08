from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from src.task10_generation import generate_with_citation
from src.task9_retrieval_pipeline import retrieve


st.set_page_config(
    page_title="RAG Pipeline Tester",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _truncate(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _source_line(chunk: dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source", "unknown")
    doc_type = metadata.get("doc_type", "-")
    score = chunk.get("score", 0.0)
    route = chunk.get("source", "hybrid")
    return f"{source} | {doc_type} | {route} | score={score:.4f}"


def _render_sources(chunks: list[dict]) -> None:
    if not chunks:
        st.info("No retrieved chunks.")
        return
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"{i}. {_source_line(chunk)}", expanded=i == 1):
            st.write(_truncate(chunk.get("content", ""), 1600))
            metadata = chunk.get("metadata", {}) or {}
            if metadata:
                st.json(metadata, expanded=False)


def _diagnostics() -> dict[str, str | int | bool]:
    chroma_count: int | str = "not checked"
    chroma_exists = (ROOT_DIR / "data" / "chroma_db").exists()
    try:
        import chromadb

        collection = chromadb.PersistentClient(
            path=str(ROOT_DIR / "data" / "chroma_db")
        ).get_or_create_collection("rag_documents")
        chroma_count = collection.count()
    except Exception as exc:
        chroma_count = f"error: {exc}"

    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "PAGEINDEX_API_KEY": bool(os.getenv("PAGEINDEX_API_KEY")),
        "PAGEINDEX_DOC_IDS": bool(os.getenv("PAGEINDEX_DOC_IDS")),
        "data/chroma_db exists": chroma_exists,
        "Chroma collection count": chroma_count,
        "standardized markdown files": len(
            list((ROOT_DIR / "data" / "standardized").glob("**/*.md"))
        ),
    }


st.title("RAG Pipeline Tester")

with st.sidebar:
    st.header("Controls")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    score_threshold = st.slider(
        "Fallback threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    mode = st.radio(
        "Mode",
        ["Generate answer", "Retrieve only"],
        index=0,
    )
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    st.caption("Example queries")
    examples = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
        "Nghệ sĩ nào bị bắt vì liên quan đến ma túy?",
        "Luật phòng chống ma túy 2021 quy định gì về cai nghiện?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_query"] = example

tab_chat, tab_retrieval, tab_diagnostics = st.tabs(
    ["Chat", "Retrieval", "Diagnostics"]
)

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_query = st.session_state.pop("pending_query", None)
    query = pending_query or st.chat_input("Ask about Vietnamese drug law or related news")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Running retrieval pipeline..."):
                chunks = retrieve(
                    query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

            if mode == "Retrieve only":
                answer = f"Retrieved {len(chunks)} chunks."
                st.markdown(answer)
            else:
                with st.spinner("Generating cited answer..."):
                    result = generate_with_citation(query, context_chunks=chunks)
                answer = result["answer"]
                st.markdown(answer)
                st.caption(
                    f"Retrieval route: {result.get('retrieval_source', 'unknown')} | "
                    f"Sources: {', '.join(result.get('sources', [])) or 'none'}"
                )

            if show_sources:
                st.subheader("Source chunks")
                _render_sources(chunks)

        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_retrieval:
    st.subheader("Inspect retrieval output")
    retrieval_query = st.text_input(
        "Query",
        value="Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
        key="retrieval_query",
    )
    if st.button("Run retrieval", type="primary"):
        with st.spinner("Retrieving..."):
            chunks = retrieve(
                retrieval_query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        st.write(f"Returned {len(chunks)} chunks")
        _render_sources(chunks)

with tab_diagnostics:
    st.subheader("Environment and data")
    st.json(_diagnostics())
    st.caption(
        "PageIndex may show no results if the account has reached LimitReached or "
        "InsufficientCredits. The hybrid path can still be tested with Chroma/BM25."
    )
