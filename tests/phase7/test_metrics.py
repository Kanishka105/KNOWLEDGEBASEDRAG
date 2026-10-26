
import pytest

from app.ai.rag.phase7_advanced_retrieval.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "D"}

    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "D", "F"}

    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    retrieved = ["A", "B", "C"]
    relevant = {"C"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_without_relevant_result():
    assert reciprocal_rank(["A", "B"], {"C"}) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "B"}

    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)

    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, k=0)

    with pytest.raises(ValueError):
        ndcg_at_k(["A"], {"A"}, k=0)
