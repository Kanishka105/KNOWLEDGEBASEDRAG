
import json
from pathlib import Path

import numpy as np


def save_vector_store(
    output_dir: str | Path,
    document_ids: list[str],
    embeddings: np.ndarray,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)

    with (output_dir / "document_ids.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(document_ids, file)


def load_vector_store(output_dir: str | Path):
    output_dir = Path(output_dir)

    embeddings = np.load(
        output_dir / "embeddings.npy",
        allow_pickle=False,
    )

    with (output_dir / "document_ids.json").open(
        "r", encoding="utf-8"
    ) as file:
        document_ids = json.load(file)

    if len(document_ids) != len(embeddings):
        raise ValueError("Document IDs and embeddings do not match.")

    return document_ids, embeddings


def search_vectors(
    query_embedding: np.ndarray,
    document_ids: list[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if len(document_ids) != len(embeddings):
        raise ValueError("Document IDs and embeddings do not match.")

    if len(document_ids) == 0:
        return []

    query_embedding = np.asarray(query_embedding).reshape(-1)

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError("Query and document embedding dimensions differ.")

    # Embeddings are normalized, so dot product equals cosine similarity.
    scores = embeddings @ query_embedding

    ranked_positions = np.argsort(-scores)[:top_k]

    return [
        {
            "rank": rank,
            "document_id": document_ids[position],
            "score": float(scores[position]),
        }
        for rank, position in enumerate(ranked_positions, start=1)
    ]
