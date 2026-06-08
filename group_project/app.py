from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation


st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLES = [
    "Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
    "Luật phòng chống ma túy 2021 quy định gì về cai nghiện?",
    "Nghệ sĩ nào bị bắt vì liên quan đến ma túy?",
]


def _truncate(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _memory_window(messages: list[dict], turns: int) -> list[dict]:
    if turns <= 0:
        return []
    return messages[-turns * 2:]


def build_contextual_query(query: str, messages: list[dict], turns: int = 3) -> str:
    """
    Add recent conversation turns to support follow-up questions.

    Retrieval still uses Task 9, but the query contains short chat memory so
    references like "ý trên", "người đó", "theo luật này" have enough context.
    """
    history = _memory_window(messages, turns)
    if not history:
        return query

    lines = ["Recent conversation for resolving follow-up references:"]
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        content = re.sub(r"\s+", " ", message["content"]).strip()
        lines.append(f"{role}: {_truncate(content, 500)}")
    lines.append(f"Current question: {query}")
    return "\n".join(lines)


def _chunk_title(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source", "unknown")
    doc_type = metadata.get("doc_type", "-")
    route = chunk.get("source", "hybrid")
    score = chunk.get("score", 0.0)
    return f"{index}. {source} | {doc_type} | {route} | score={score:.4f}"


def render_sources(chunks: list[dict]) -> None:
    if not chunks:
        st.info("Không có source documents.")
        return

    for index, chunk in enumerate(chunks, 1):
        with st.expander(_chunk_title(chunk, index), expanded=index == 1):
            st.markdown(_truncate(chunk.get("content", "")))
            metadata = chunk.get("metadata", {}) or {}
            if metadata:
                st.json(metadata, expanded=False)


def diagnostics() -> dict:
    chroma_count: int | str = "not checked"
    try:
        import chromadb

        collection = chromadb.PersistentClient(
            path=str(ROOT_DIR / "data" / "chroma_db")
        ).get_or_create_collection("rag_documents")
        chroma_count = collection.count()
    except Exception as exc:
        chroma_count = f"error: {exc}"

    return {
        "OPENAI_API_KEY set": bool(os.getenv("OPENAI_API_KEY")),
        "PAGEINDEX_API_KEY set": bool(os.getenv("PAGEINDEX_API_KEY")),
        "PAGEINDEX_DOC_IDS set": bool(os.getenv("PAGEINDEX_DOC_IDS")),
        "Chroma chunks": chroma_count,
        "Markdown files": len(list((ROOT_DIR / "data" / "standardized").glob("**/*.md"))),
    }


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_sources = []


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

st.title("Drug Law RAG Chatbot")
st.caption("Streamlit -> Task 9 Retrieval -> Task 10 Generation with citation")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K sources", 1, 10, 5)
    score_threshold = st.slider("PageIndex fallback threshold", 0.0, 1.0, 0.3, 0.05)
    memory_turns = st.slider("Conversation memory turns", 0, 5, 3)
    show_contextual_query = st.toggle("Show contextualized query", value=False)
    show_sources = st.toggle("Show source documents", value=True)

    st.divider()
    st.button("Clear chat", on_click=reset_chat, use_container_width=True)

    st.divider()
    st.subheader("Examples")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    st.subheader("Diagnostics")
    st.json(diagnostics(), expanded=False)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending_query = st.session_state.pop("pending_query", None)
query = pending_query or st.chat_input("Hỏi về pháp luật ma túy hoặc tin tức liên quan")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    contextual_query = build_contextual_query(
        query,
        st.session_state.messages[:-1],
        turns=memory_turns,
    )

    with st.chat_message("assistant"):
        if show_contextual_query and contextual_query != query:
            with st.expander("Contextualized query"):
                st.code(contextual_query)

        with st.spinner("Retrieving relevant documents..."):
            chunks = retrieve(
                contextual_query,
                top_k=top_k,
                score_threshold=score_threshold,
            )

        with st.spinner("Generating cited answer..."):
            result = generate_with_citation(query, context_chunks=chunks)

        answer = result["answer"]
        st.markdown(answer)
        st.caption(
            f"Retrieval route: {result.get('retrieval_source', 'none')} | "
            f"Sources: {', '.join(result.get('sources', [])) or 'none'}"
        )

        if show_sources:
            st.subheader("Source documents")
            render_sources(chunks)

    st.session_state.last_sources = chunks
    st.session_state.messages.append({"role": "assistant", "content": answer})
