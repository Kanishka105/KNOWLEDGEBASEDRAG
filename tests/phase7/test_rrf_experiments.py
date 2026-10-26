
import pytest

from app.ai.rag.phase3_hybrid_retrieval.hybrid_retrieval import (
    reciprocal_rank_fusion,
)


def test_rrf_returns_scores_for_all_unique_documents():
    rankings = [
        ["A", "B", "C"],
        ["B", "C", "D"],
    ]

    scores = reciprocal_rank_fusion(rankings, k=60)

    assert set(scores) == {"A", "B", "C", "D"}
    assert scores["B"] > scores["A"]


@pytest.mark.parametrize("rrf_k", [20, 40, 60, 100])
def test_rrf_accepts_candidate_constants(rrf_k):
    rankings = [
        ["A", "B", "C"],
        ["B", "C", "A"],
    ]

    scores = reciprocal_rank_fusion(rankings, k=rrf_k)

    assert scores
    assert all(score > 0 for score in scores.values())


def test_rrf_rejects_invalid_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["A"]], k=0)
