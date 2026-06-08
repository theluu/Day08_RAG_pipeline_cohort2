"""
RAG Evaluation Pipeline for the group project.

Framework choice: custom deterministic evaluator.

Reason:
  DeepEval/RAGAS/TruLens are strong options, but they require extra packages and
  usually LLM-as-judge API calls. This script keeps the required 4 metrics,
  produces repeatable scores, and runs with the existing RAG modules.

Metrics implemented:
  - Faithfulness: answer tokens covered by retrieved context tokens.
  - Answer Relevance: expected-answer tokens covered by actual answer tokens.
  - Context Recall: expected context hints found in retrieved context.
  - Context Precision: fraction of retrieved chunks containing expected hints.

A/B configs:
  - hybrid_rerank: Task 9 full pipeline + Task 10 generation.
  - lexical_only: Task 6 BM25 baseline + Task 10 generation.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
DEFAULT_TOP_K = 5

VI_STOPWORDS = {
    "và", "là", "của", "có", "cho", "theo", "về", "trong", "những", "các",
    "một", "được", "nào", "gì", "như", "nêu", "bài", "viết", "quy", "định",
    "với", "từ", "đến", "ở", "bị", "vì", "liên", "quan", "the", "and", "or",
}


@dataclass
class EvalRow:
    config: str
    case_id: str
    question: str
    answer: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    context_precision: float
    average: float
    retrieved_sources: list[str]
    failure_stage: str
    root_cause: str


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) < 15:
        raise ValueError(f"Golden dataset must have >=15 cases, got {len(data)}")
    return data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", normalize_text(text), flags=re.UNICODE)
    return {token for token in tokens if len(token) > 2 and token not in VI_STOPWORDS}


def contains_hint(text: str, hint: str) -> bool:
    text_norm = normalize_text(text)
    hint_norm = normalize_text(hint)
    if hint_norm in text_norm:
        return True

    hint_tokens = tokenize(hint)
    if not hint_tokens:
        return False
    text_tokens = tokenize(text)
    return len(hint_tokens & text_tokens) / len(hint_tokens) >= 0.6


def ratio_overlap(source_text: str, target_text: str) -> float:
    source_tokens = tokenize(source_text)
    target_tokens = tokenize(target_text)
    if not source_tokens:
        return 0.0
    return round(len(source_tokens & target_tokens) / len(source_tokens), 4)


def context_text(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        parts.append(str(metadata.get("source", "")))
        parts.append(chunk.get("content", ""))
    return "\n".join(parts)


def score_context_recall(expected_hints: list[str], chunks: list[dict]) -> float:
    if not expected_hints:
        return 0.0
    ctx = context_text(chunks)
    matched = sum(1 for hint in expected_hints if contains_hint(ctx, hint))
    return round(matched / len(expected_hints), 4)


def score_context_precision(expected_hints: list[str], chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    useful = 0
    for chunk in chunks:
        chunk_text = context_text([chunk])
        if any(contains_hint(chunk_text, hint) for hint in expected_hints):
            useful += 1
    return round(useful / len(chunks), 4)


def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    if answer.strip() == "I cannot verify this information":
        return 0.0
    return ratio_overlap(answer, context_text(chunks))


def score_answer_relevance(expected_answer: str, answer: str) -> float:
    if answer.strip() == "I cannot verify this information":
        return 0.0
    return ratio_overlap(expected_answer, answer)


def classify_failure(row: EvalRow) -> tuple[str, str]:
    if not row.retrieved_sources:
        return "retrieval", "No context returned"
    if row.context_recall < 0.5:
        return "retrieval", "Retriever missed expected evidence"
    if row.context_precision < 0.4:
        return "retrieval", "Retrieved context has too much unrelated material"
    if row.answer_relevance < 0.4:
        return "generation", "Answer does not cover expected answer"
    if row.faithfulness < 0.5:
        return "generation", "Answer terms are weakly grounded in retrieved context"
    return "ok", "Meets heuristic thresholds"


def hybrid_rerank_pipeline(question: str, top_k: int = DEFAULT_TOP_K) -> tuple[str, list[dict]]:
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    chunks = retrieve(question, top_k=top_k)
    result = generate_with_citation(question, context_chunks=chunks)
    return result["answer"], chunks


def lexical_only_pipeline(question: str, top_k: int = DEFAULT_TOP_K) -> tuple[str, list[dict]]:
    from src.task6_lexical_search import lexical_search
    from src.task10_generation import generate_with_citation

    chunks = lexical_search(question, top_k=top_k)
    chunks = [{**chunk, "source": "lexical_only"} for chunk in chunks]
    result = generate_with_citation(question, context_chunks=chunks)
    return result["answer"], chunks


def evaluate_config(
    config_name: str,
    pipeline: Callable[[str, int], tuple[str, list[dict]]],
    golden_dataset: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> list[EvalRow]:
    rows = []
    for item in golden_dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_context = item["expected_context"]

        try:
            answer, chunks = pipeline(question, top_k)
        except Exception as exc:
            answer, chunks = f"ERROR: {exc}", []

        faithfulness = score_faithfulness(answer, chunks)
        answer_relevance = score_answer_relevance(expected_answer, answer)
        context_recall = score_context_recall(expected_context, chunks)
        context_precision = score_context_precision(expected_context, chunks)
        average = round(statistics.mean([
            faithfulness,
            answer_relevance,
            context_recall,
            context_precision,
        ]), 4)
        sources = [
            (chunk.get("metadata", {}) or {}).get("source", "unknown")
            for chunk in chunks
        ]

        row = EvalRow(
            config=config_name,
            case_id=item["id"],
            question=question,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_recall=context_recall,
            context_precision=context_precision,
            average=average,
            retrieved_sources=sources,
            failure_stage="",
            root_cause="",
        )
        row.failure_stage, row.root_cause = classify_failure(row)
        rows.append(row)
        print(
            f"[{config_name}] {row.case_id}: avg={row.average:.3f} "
            f"recall={row.context_recall:.3f} precision={row.context_precision:.3f}"
        )
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, float]:
    return {
        "faithfulness": round(statistics.mean(r.faithfulness for r in rows), 4),
        "answer_relevance": round(statistics.mean(r.answer_relevance for r in rows), 4),
        "context_recall": round(statistics.mean(r.context_recall for r in rows), 4),
        "context_precision": round(statistics.mean(r.context_precision for r in rows), 4),
        "average": round(statistics.mean(r.average for r in rows), 4),
    }


def compare_configs(golden_dataset: list[dict]) -> dict:
    configs = {
        "hybrid_rerank": hybrid_rerank_pipeline,
        "lexical_only": lexical_only_pipeline,
    }
    all_rows = {}
    summaries = {}
    for name, pipeline in configs.items():
        rows = evaluate_config(name, pipeline, golden_dataset)
        all_rows[name] = rows
        summaries[name] = summarize(rows)
    return {"rows": all_rows, "summaries": summaries}


def metric_row(label: str, key: str, summaries: dict) -> str:
    a = summaries["hybrid_rerank"][key]
    b = summaries["lexical_only"][key]
    delta = round(a - b, 4)
    return f"| {label} | {a:.4f} | {b:.4f} | {delta:+.4f} |"


def export_results(comparison: dict) -> None:
    summaries = comparison["summaries"]
    rows_by_config = comparison["rows"]
    hybrid_rows = rows_by_config["hybrid_rerank"]
    worst = sorted(hybrid_rows, key=lambda row: row.average)[:3]
    winner = "hybrid_rerank" if summaries["hybrid_rerank"]["average"] >= summaries["lexical_only"]["average"] else "lexical_only"

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "Custom deterministic evaluator, implemented in `group_project/evaluation/eval_pipeline.py`.",
        "The evaluator measures the four required RAG dimensions without extra judge dependencies.",
        "",
        "## Dataset",
        "",
        f"- Golden dataset size: {len(hybrid_rows)} Q&A pairs",
        "- Domain: Vietnamese drug law and related news",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A: hybrid + rerank | Config B: lexical-only | Δ |",
        "|--------|---------------------------|------------------------|---|",
        metric_row("Faithfulness", "faithfulness", summaries),
        metric_row("Answer Relevance", "answer_relevance", summaries),
        metric_row("Context Recall", "context_recall", summaries),
        metric_row("Context Precision", "context_precision", summaries),
        metric_row("Average", "average", summaries),
        "",
        "## A/B Comparison Analysis",
        "",
        "**Config A: hybrid_rerank**",
        "",
        "Uses Task 9 full retrieval: semantic search + BM25 lexical search, RRF merge, rerank, and PageIndex fallback when available.",
        "",
        "**Config B: lexical_only**",
        "",
        "Uses only Task 6 BM25 chunks, then the same Task 10 citation generator. This isolates the value of hybrid retrieval and reranking.",
        "",
        f"**Kết luận:** `{winner}` has the higher average score in this run. "
        "Use the per-case table below to inspect whether errors come from retrieval coverage or generation grounding.",
        "",
        "## Worst Performers (Bottom 3, Config A)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |",
        "|---|----------|--------------|-----------|--------|-----------|---------------|------------|",
    ]
    for index, row in enumerate(worst, 1):
        question = row.question.replace("|", "/")
        lines.append(
            f"| {index} | {question} | {row.faithfulness:.4f} | "
            f"{row.answer_relevance:.4f} | {row.context_recall:.4f} | "
            f"{row.context_precision:.4f} | {row.failure_stage} | {row.root_cause} |"
        )

    lines.extend([
        "",
        "## Per-Case Scores",
        "",
        "| Config | ID | Avg | Faithfulness | Relevance | Recall | Precision | Sources |",
        "|--------|----|-----|--------------|-----------|--------|-----------|---------|",
    ])
    for config_name, rows in rows_by_config.items():
        for row in rows:
            sources = ", ".join(dict.fromkeys(row.retrieved_sources[:3]))
            lines.append(
                f"| {config_name} | {row.case_id} | {row.average:.4f} | "
                f"{row.faithfulness:.4f} | {row.answer_relevance:.4f} | "
                f"{row.context_recall:.4f} | {row.context_precision:.4f} | {sources} |"
            )

    lines.extend([
        "",
        "## Recommendations",
        "",
        "### Cải tiến 1",
        "**Action:** Clean markdown boilerplate from crawled news pages before chunking.",
        "**Expected impact:** Higher context precision and fewer irrelevant chunks.",
        "",
        "### Cải tiến 2",
        "**Action:** Add metadata fields such as publication date, source platform, and legal article number during conversion.",
        "**Expected impact:** Better citation quality and easier source filtering.",
        "",
        "### Cải tiến 3",
        "**Action:** Increase PageIndex credits or provide `PAGEINDEX_DOC_IDS` for uploaded legal PDFs.",
        "**Expected impact:** Better fallback coverage when hybrid retrieval score is weak.",
        "",
    ])

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
