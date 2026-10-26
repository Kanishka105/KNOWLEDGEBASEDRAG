
import math


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Fraction of the top-k retrieved results that are relevant."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not retrieved_ids[:k]:
        return 0.0

    hits = sum(
        1 for item in retrieved_ids[:k]
        if item in relevant_ids
    )

    # Standard Precision@K uses k as its denominator.
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Fraction of all known relevant items retrieved in the top-k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_ids:
        return 0.0

    hits = sum(
        1 for item in retrieved_ids[:k]
        if item in relevant_ids
    )

    return hits / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    actual_dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, item in enumerate(retrieved_ids[:k], start=1)
        if item in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg
