
import json
from pathlib import Path

import faiss
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[4]
VECTOR_DIR = ROOT_DIR / "outputs" / "phase2" / "vector_store"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_faiss_index():
    """Load Phase 2 embeddings and build an exact FAISS index."""

    embeddings_path = VECTOR_DIR / "embeddings.npy"
    ids_path = VECTOR_DIR / "document_ids.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}"
        )

    if not ids_path.exists():
        raise FileNotFoundError(
            f"Document IDs file not found: {ids_path}"
        )

    embeddings = np.load(embeddings_path).astype(np.float32)

    document_ids = json.loads(
        ids_path.read_text(encoding="utf-8")
    )
    document_ids = [str(doc_id) for doc_id in document_ids]

    if embeddings.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix.")

    if embeddings.shape[0] != len(document_ids):
        raise ValueError("Embedding rows and document IDs do not match.")

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate document IDs detected.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    # Cosine similarity via inner product requires normalized vectors.
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Stored embeddings are not normalized. Check Phase 2."
        )

    dimension = embeddings.shape[1]

    # Exact inner-product index.
    # Since vectors are normalized, this performs cosine-similarity search.
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(embeddings))

    return index, document_ids


def search_faiss(
    query_embedding,
    index,
    document_ids,
    top_k=5,
):
    """Return the nearest document IDs and cosine similarity scores."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"Query dimension {query_vector.shape[1]} "
            f"does not match index dimension {index.d}."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values.")

    if np.linalg.norm(query_vector) == 0:
        raise ValueError("Query embedding must not be a zero vector.")

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        np.ascontiguousarray(query_vector),
        min(top_k, index.ntotal),
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            {
                "document_id": document_ids[int(idx)],
                "cosine_similarity": float(score),
            }
        )

    return results


def main():
    from sentence_transformers import SentenceTransformer

    print("Loading Phase 2 embeddings and building FAISS index...")

    index, document_ids = build_faiss_index()

    print("FAISS index built successfully.")
    print("Documents indexed:", index.ntotal)
    print("Vector dimension:", index.d)

    model = SentenceTransformer(MODEL_NAME)

    query = "0-dimensional biomaterials show inductive properties."

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    results = search_faiss(
        query_embedding=query_embedding,
        index=index,
        document_ids=document_ids,
        top_k=5,
    )

    print("\nQuery:", query)
    print("\nTop 5 FAISS results:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. Document ID: {result['document_id']} | "
            f"Cosine similarity: "
            f"{result['cosine_similarity']:.4f}"
        )


if __name__ == "__main__":
    main()
