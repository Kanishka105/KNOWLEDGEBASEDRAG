

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase1_dataset.text_processing import (
    normalize_document_text,
    tokenize,
)

ROOT_DIR = Path(__file__).resolve().parents[4]

VECTOR_DIR = ROOT_DIR / "outputs" / "phase2" / "vector_store"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase3"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
CANDIDATE_K = 100
RERANK_K = 50

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Combine ranked document lists using Reciprocal Rank Fusion.

    Each document receives a score of 1 / (k + rank)
    for each ranking in which it appears. Ranks start at 1.
    """
    if k < 1:
        raise ValueError("RRF k must be at least 1.")

    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        seen: set[str] = set()

        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in seen:
                continue

            seen.add(doc_id)

            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return fused_scores

def retrieve_hybrid(
    query: str,
    corpus: dict,
    embedding_model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    document_ids: list[str],
    bm25_index: BM25Okapi,
    bm25_ids: list[str],
    faiss_index,
    top_k: int = 5,
    candidate_k: int = CANDIDATE_K,
    rerank_k: int = RERANK_K,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Hybrid retrieval pipeline:

    1. Retrieve candidates using BM25.
    2. Retrieve candidates using FAISS dense search.
    3. Fuse rankings using Reciprocal Rank Fusion (RRF).
    4. Rerank fused candidates using a CrossEncoder.
    5. Return the final top-k documents.

    Args:
        query: User query or scientific claim.
        corpus: Dictionary mapping document IDs to documents.
        embedding_model: Loaded SentenceTransformer model.
        cross_encoder: Loaded CrossEncoder reranker.
        document_ids: Document IDs aligned with FAISS vector positions.
        bm25_index: BM25 index.
        bm25_ids: Document IDs aligned with BM25 corpus positions.
        faiss_index: Built FAISS index.
        top_k: Number of final results.
        candidate_k: Maximum candidates retrieved from each retriever.
        rerank_k: Maximum fused candidates sent to the CrossEncoder.
        rrf_k: RRF rank constant. Defaults to the original value of 60.

    Returns:
        Ranked documents containing IDs, scores, titles, and text.
    """
    query = extract_query_text(query)

    if top_k < 1 or candidate_k < 1 or rerank_k < 1:
        raise ValueError(
            "top_k, candidate_k, and rerank_k must be positive."
        )

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive.")

    if not document_ids:
        raise ValueError("FAISS document IDs cannot be empty.")

    if not bm25_ids:
        raise ValueError("BM25 document IDs cannot be empty.")

    if len(bm25_ids) != len(corpus):
        raise ValueError(
            "BM25 document IDs must match the corpus size."
        )

    if len(document_ids) != faiss_index.ntotal:
        raise ValueError(
            "FAISS document IDs must match the FAISS index size."
        )

    print(f"Running BM25 retrieval for: {query}")

    # --------------------------------------------------------
    # STEP 1: BM25 SPARSE RETRIEVAL
    # --------------------------------------------------------
    bm25_scores = bm25_index.get_scores(tokenize(query))

    bm25_limit = min(candidate_k, len(bm25_ids))
    bm25_order = np.argsort(-bm25_scores)[:bm25_limit]

    bm25_ranked_ids = [
        bm25_ids[int(position)]
        for position in bm25_order
    ]

    print(f"BM25 candidates: {len(bm25_ranked_ids)}")

    # --------------------------------------------------------
    # STEP 2: FAISS DENSE RETRIEVAL
    # --------------------------------------------------------
    print("Running FAISS semantic retrieval...")

    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if query_vector.ndim != 2:
        raise ValueError("Query embedding must be a 2D matrix.")

    if query_vector.shape[1] != faiss_index.d:
        raise ValueError(
            "Query embedding dimension does not match FAISS index."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError(
            "Query embedding contains invalid numeric values."
        )

    dense_limit = min(candidate_k, faiss_index.ntotal)

    faiss_scores, faiss_positions = faiss_index.search(
        np.ascontiguousarray(query_vector),
        dense_limit,
    )

    dense_ranked_ids = [
        document_ids[int(position)]
        for position in faiss_positions[0]
        if position >= 0
    ]

    dense_scores = {
        document_ids[int(position)]: float(score)
        for position, score in zip(
            faiss_positions[0],
            faiss_scores[0],
        )
        if position >= 0
    }

    print(f"FAISS candidates: {len(dense_ranked_ids)}")

    # --------------------------------------------------------
    # STEP 3: RECIPROCAL RANK FUSION
    # --------------------------------------------------------
    print(f"Combining rankings using RRF (k={rrf_k})...")

    fused_scores = reciprocal_rank_fusion(
        [
            bm25_ranked_ids,
            dense_ranked_ids,
        ],
        k=rrf_k,
    )

    rrf_candidate_ids = sorted(
        fused_scores,
        key=lambda doc_id: (
            -fused_scores[doc_id],
            doc_id,
        ),
    )[:min(rerank_k, len(fused_scores))]

    print(f"Unique fused candidates: {len(fused_scores)}")
    print(f"Candidates for reranking: {len(rrf_candidate_ids)}")

    # --------------------------------------------------------
    # STEP 4: CROSSENCODER RERANKING
    # --------------------------------------------------------
    print("Reranking candidates using CrossEncoder...")

    pairs = [
        (
            query,
            normalize_document_text(corpus[doc_id]),
        )
        for doc_id in rrf_candidate_ids
    ]

    if not pairs:
        return []

    cross_scores = cross_encoder.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    cross_scores = np.asarray(cross_scores).reshape(-1)

    if len(cross_scores) != len(rrf_candidate_ids):
        raise ValueError(
            "CrossEncoder returned an unexpected number of scores."
        )

    if not np.isfinite(cross_scores).all():
        raise ValueError(
            "CrossEncoder returned invalid scores."
        )

    bm25_ranks = {
        doc_id: rank
        for rank, doc_id in enumerate(bm25_ranked_ids, start=1)
    }

    dense_ranks = {
        doc_id: rank
        for rank, doc_id in enumerate(dense_ranked_ids, start=1)
    }

    reranked = sorted(
        zip(rrf_candidate_ids, cross_scores),
        key=lambda item: (
            -float(item[1]),
            item[0],
        ),
    )

    # --------------------------------------------------------
    # STEP 5: PREPARE FINAL RESULTS
    # --------------------------------------------------------
    results = []

    for rank, (doc_id, cross_score) in enumerate(
        reranked[:top_k],
        start=1,
    ):
        document = corpus[doc_id]

        results.append({
            "rank": rank,
            "document_id": doc_id,
            "title": document.get("title", ""),
            "hybrid_score": float(fused_scores[doc_id]),
            "cross_encoder_score": float(cross_score),
            "bm25_rank": bm25_ranks.get(doc_id),
            "dense_rank": dense_ranks.get(doc_id),
            "dense_similarity": dense_scores.get(doc_id),
            "text": normalize_document_text(document),
        })

    return results
