from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from rag_pipeline import embed_text, retrieve
from rag_observability import evaluate_rag, rerank


def test_embedding_is_custom_384_dimension_hash_vector() -> None:
    vector = embed_text("cosine similarity and retrieval")
    assert len(vector) == 384
    assert abs(sum(value * value for value in vector) - 1.0) < 0.001


def test_retrieve_combines_lexical_and_semantic_scores() -> None:
    rows = [
        {"chunk_id": "exact", "chunk_text": "A cognitive load assessment measures mental effort.", "vector_similarity": 0.91},
        {"chunk_id": "other", "chunk_text": "A participant completes a visual attention task.", "vector_similarity": 0.12},
    ]
    results = retrieve("cognitive load assessment", rows, limit=2)
    assert results[0]["chunk_id"] == "exact"
    assert results[0]["lexical_similarity"] > 0
    assert results[0]["semantic_similarity"] > 0
    assert results[0]["similarity"] > 0


def test_reranker_adds_a_distinct_final_score() -> None:
    rows = [{
        "chunk_id": "relevant",
        "chunk_text": "PERCLOS contributes to a fatigue-related eye metric.",
        "similarity": 0.62,
    }]
    result = rerank("How is PERCLOS used for fatigue?", rows, limit=1)[0]
    assert result["chunk_id"] == "relevant"
    assert result["rerank_score"] > 0
    assert result["rerank_query_overlap"] > 0


def test_evaluation_records_core_metrics_without_fabricating_recall() -> None:
    result = evaluate_rag(
        "What does PERCLOS measure?",
        "PERCLOS measures the proportion of recent samples with closed eyes.",
        ["PERCLOS is the proportion of recent valid samples in which the eyes were closed."],
        [0.86],
    )
    assert result["evaluator"] == "native"
    assert result["evaluation_status"] == "complete"
    assert result["context_precision"] > 0
    assert result["faithfulness"] > 0
    assert result["answer_relevancy"] > 0
    assert result["context_recall"] == "unavailable"
