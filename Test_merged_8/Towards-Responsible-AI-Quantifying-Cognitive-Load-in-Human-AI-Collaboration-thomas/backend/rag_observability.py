"""RAG reranking, evaluation, and trace helpers.

The default evaluator is dependency-light and deterministic. It exposes the
same core dimensions commonly used by RAGAS/DeepEval (context precision,
context recall, faithfulness, and answer relevancy) without pretending that a
third-party evaluator ran when it was not installed. Optional RAGAS/DeepEval
adapters can be selected explicitly through configuration later.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "what", "when", "where", "which", "why", "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if token.lower() not in STOPWORDS and len(token) > 1
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return round(len(left & right) / len(left), 4)


def new_trace_id() -> str:
    return f"TRACE-{uuid.uuid4().hex}"


def rerank(query: str, candidates: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    """Rerank retrieved candidates using query/document feature matching.

    This is deliberately called a feature reranker, not a neural cross
    encoder. It is deterministic, local, and available without downloading a
    model. A future CrossEncoder can replace this function behind the same
    contract without changing the API or trace schema.
    """
    query_tokens = _tokens(query)
    reranked: list[dict[str, Any]] = []
    for row in candidates:
        text = str(row.get("chunk_text") or "")
        text_tokens = _tokens(text)
        lexical_focus = _overlap(query_tokens, text_tokens)
        hybrid_score = float(row.get("similarity") or 0.0)
        # The reranker rewards direct query-term coverage while retaining the
        # retriever's hybrid score as the dominant signal.
        rerank_score = round((0.65 * hybrid_score) + (0.35 * lexical_focus), 4)
        reranked.append({
            **row,
            "rerank_score": rerank_score,
            "rerank_query_overlap": lexical_focus,
        })
    reranked.sort(
        key=lambda row: (float(row.get("rerank_score") or 0.0), float(row.get("similarity") or 0.0)),
        reverse=True,
    )
    return reranked[:max(1, limit)]


def evaluate_rag(
    query: str,
    answer: str,
    contexts: list[str],
    retrieved_scores: list[float],
    reference_answer: str = "",
    evaluator: str = "native",
) -> dict[str, Any]:
    """Calculate auditable, local RAG quality metrics.

    If a reference answer is supplied, context recall and answer relevancy
    are measured against it. Without a reference, recall is explicitly marked
    unavailable rather than fabricated.
    """
    query_tokens = _tokens(query)
    answer_tokens = _tokens(answer)
    context_tokens = [_tokens(context) for context in contexts if context]
    relevant_context = [_overlap(query_tokens, tokens) for tokens in context_tokens]
    context_precision = round(
        sum(score for score in relevant_context if score > 0) / len(relevant_context), 4
    ) if relevant_context else 0.0
    answer_context_overlap = max((_overlap(answer_tokens, tokens) for tokens in context_tokens), default=0.0)
    answer_relevancy = _overlap(query_tokens, answer_tokens)
    context_recall: float | None = None
    if reference_answer:
        reference_tokens = _tokens(reference_answer)
        context_recall = round(
            max((_overlap(reference_tokens, tokens) for tokens in context_tokens), default=0.0), 4
        )
        answer_relevancy = _overlap(reference_tokens, answer_tokens)
    faithfulness = answer_context_overlap
    return {
        "evaluator": evaluator,
        "evaluation_status": "complete",
        "context_precision": context_precision,
        "context_recall": context_recall if context_recall is not None else "unavailable",
        "faithfulness": faithfulness,
        "answer_faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "mean_retrieval_score": round(sum(retrieved_scores) / len(retrieved_scores), 4) if retrieved_scores else 0.0,
        "evaluated_context_count": len(contexts),
    }


def timed() -> tuple[float, float]:
    """Return a monotonic start time and wall-clock timestamp."""
    return time.perf_counter(), time.time()
