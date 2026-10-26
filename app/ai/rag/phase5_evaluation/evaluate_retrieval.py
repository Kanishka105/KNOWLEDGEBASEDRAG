
import argparse
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
from app.ai.rag.phase3_hybrid_retrieval.hybrid_retrieval import (
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_NAME,
    CANDIDATE_K,
    RRF_K,
    build_hybrid_index,
    reciprocal_rank_fusion,
    retrieve_hybrid,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[4]
VECTOR_DIR = ROOT_DIR / "outputs" / "phase2" / "vector_store"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase5"

TOP_K = 5


# ============================================================
# QUERY AND RELEVANCE NORMALIZATION
# ============================================================

def get_query_text(query_data) -> str:
    """Extract query text from supported query structures."""
    if isinstance(query_data, str):
        text = query_data

    elif isinstance(query_data, dict):
        text = next(
            (
                query_data[key]
                for key in ("text", "query", "query_text")
                if isinstance(query_data.get(key), str)
            ),
            None,
        )

        if text is None:
            raise ValueError(
                f"Cannot find query text in fields: "
                f"{list(query_data.keys())}"
            )
    else:
        raise TypeError(
            f"Unsupported query type: {type(query_data).__name__}"
        )

    text = text.strip()

    if not text:
        raise ValueError("Encountered an empty query.")

    return text


def get_relevant_ids(query_qrels) -> set[str]:
    """Convert supported relevance judgments to document ID sets."""
    relevant = set()

    if isinstance(query_qrels, dict):
        for doc_id, score in query_qrels.items():
            try:
                is_relevant = float(score) > 0
            except (TypeError, ValueError):
                is_relevant = bool(score)

            if is_relevant:
                relevant.add(str(doc_id))

    elif isinstance(query_qrels, (list, tuple, set)):
        for item in query_qrels:
            if isinstance(item, dict):
                doc_id = item.get(
                    "doc_id",
                    item.get(
                        "document_id",
                        item.get("corpus_id"),
                    ),
                )

                score = item.get(
                    "relevance",
                    item.get("score", item.get("label", 1)),
                )

                if doc_id is not None and float(score) > 0:
                    relevant.add(str(doc_id))
            else:
                relevant.add(str(item))

    else:
        raise TypeError(
            f"Unsupported qrels structure: {type(query_qrels).__name__}"
        )

    return relevant


# ============================================================
# EVALUATION METRICS
# ============================================================

def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> float:
    """Fraction of the top-k positions containing relevant documents."""
    if k <= 0:
        raise ValueError("k must be positive.")

    return sum(doc_id in relevant for doc_id in retrieved[:k]) / k


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> float:
    """Fraction of all known relevant documents retrieved in top-k."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant:
        return 0.0

    found = sum(
        doc_id in relevant
        for doc_id in retrieved[:k]
    )

    return found / len(relevant)


def reciprocal_rank_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> float:
    """Reciprocal rank of the first relevant result."""
    if k <= 0:
        raise ValueError("k must be positive.")

    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive.")

    gains = [
        1.0 if doc_id in relevant else 0.0
        for doc_id in retrieved[:k]
    ]

    dcg = sum(
        gain / np.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
    )

    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return float(dcg / idcg) if idcg > 0 else 0.0


def calculate_metrics(
    retrieved: list[str],
    relevant: set[str],
    k: int = TOP_K,
) -> dict:
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"mrr@{k}": reciprocal_rank_at_k(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data():
    corpus, queries, qrels = load_scifact(split="test")

    if not queries:
        raise ValueError("No queries were loaded.")

    if not qrels:
        raise ValueError("No relevance judgments were loaded.")

    corpus = {
        str(doc_id): document
        for doc_id, document in corpus.items()
    }

    queries = {
        str(query_id): get_query_text(query)
        for query_id, query in queries.items()
    }

    normalized_qrels = {
        str(query_id): get_relevant_ids(relevance)
        for query_id, relevance in qrels.items()
    }

    query_ids = [
        query_id
        for query_id, relevant in normalized_qrels.items()
        if query_id in queries and relevant
    ]

    if not query_ids:
        raise ValueError(
            "No queries have both text and relevant document IDs."
        )

    print(f"Corpus documents: {len(corpus)}")
    print(f"Queries loaded: {len(queries)}")
    print(f"Queries available for evaluation: {len(query_ids)}")

    return corpus, queries, normalized_qrels, query_ids


# ============================================================
# BUILD BM25 BASELINE
# ============================================================

def build_bm25(corpus: dict):
    """Build BM25 using a stable corpus ID-to-position mapping."""
    bm25_ids = list(corpus.keys())

    documents = [
        normalize_document_text(corpus[doc_id])
        for doc_id in bm25_ids
    ]

    tokenized = [
        tokenize(document)
        for document in documents
    ]

    return BM25Okapi(tokenized), bm25_ids


# ============================================================
# BUILD DENSE BASELINE
# ============================================================

def build_dense_index():
    """Load Phase 2 embeddings and construct a FAISS index."""
    embeddings_path = VECTOR_DIR / "embeddings.npy"
    ids_path = VECTOR_DIR / "document_ids.json"

    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            "Phase 2 vector-store files are missing. Run Phase 2 first."
        )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    ).astype(np.float32)

    document_ids = [
        str(doc_id)
        for doc_id in json.loads(
            ids_path.read_text(encoding="utf-8")
        )
    ]

    if embeddings.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    if embeddings.shape[0] != len(document_ids):
        raise ValueError(
            "Embedding rows do not match the document ID count."
        )

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate document IDs in vector store.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numeric values.")

    if len(document_ids) == 0 or embeddings.shape[1] == 0:
        raise ValueError("Vector store is empty.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Document vectors are not normalized.")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))

    return index, document_ids


# ============================================================
# RRF WITHOUT CROSS-ENCODER RERANKING
# ============================================================

def retrieve_rrf_only(
    query: str,
    embedding_model,
    document_ids: list[str],
    bm25_index,
    bm25_ids: list[str],
    faiss_index,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    rrf_k: int = RRF_K,
) -> list[str]:
    """
    Retrieve using BM25 + dense search + RRF.

    IMPORTANT:
    This function does not use the Cross-Encoder.
    It returns the top-k documents from the fused ranking directly.
    """
    if top_k < 1 or candidate_k < 1 or rrf_k < 1:
        raise ValueError("top_k, candidate_k, and rrf_k must be positive.")

    query = get_query_text(query)

    # 1. Retrieve lexical candidates.
    bm25_scores = bm25_index.get_scores(tokenize(query))

    bm25_limit = min(candidate_k, len(bm25_ids))

    bm25_positions = np.argsort(-bm25_scores)[:bm25_limit]

    bm25_ranked_ids = [
        bm25_ids[int(position)]
        for position in bm25_positions
    ]

    # 2. Retrieve semantic candidates.
    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if query_vector.ndim != 2:
        raise ValueError("Query embedding must be two-dimensional.")

    if query_vector.shape[1] != faiss_index.d:
        raise ValueError(
            "Query embedding dimension does not match FAISS index."
        )

    dense_limit = min(candidate_k, faiss_index.ntotal)

    _, dense_positions = faiss_index.search(
        np.ascontiguousarray(query_vector),
        dense_limit,
    )

    dense_ranked_ids = [
        document_ids[int(position)]
        for position in dense_positions[0]
        if position >= 0
    ]

    # 3. Fuse both rankings.
    fused_scores = reciprocal_rank_fusion(
        [bm25_ranked_ids, dense_ranked_ids],
        k=rrf_k,
    )

    # Deterministic tie-breaking by document ID.
    ranked_ids = sorted(
        fused_scores,
        key=lambda doc_id: (
            -fused_scores[doc_id],
            doc_id,
        ),
    )

    return ranked_ids[:top_k]


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_id_mappings(
    corpus: dict,
    dense_ids: list[str],
    hybrid_ids: list[str],
    hybrid_bm25_ids: list[str],
):
    """
    Validate mappings required by this project's current Phase 2 setup.

    Phase 2 creates embeddings in corpus iteration order, so the saved
    IDs are expected to match that order.
    """
    expected_ids = list(corpus.keys())

    if dense_ids != expected_ids:
        raise ValueError(
            "Dense vector IDs do not match the corpus order. "
            "Rebuild the Phase 2 vector store before evaluation."
        )

    if hybrid_ids != expected_ids:
        raise ValueError(
            "Hybrid vector IDs do not match the corpus order. "
            "Check Phase 3 vector loading and rebuild if necessary."
        )

    if hybrid_bm25_ids != expected_ids:
        raise ValueError(
            "Hybrid BM25 IDs do not match the evaluation corpus order."
        )

    if set(dense_ids) != set(corpus):
        raise ValueError("Dense document IDs differ from corpus IDs.")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(max_queries: int | None = 30):
    """
    Compare four retrieval configurations:

    1. BM25
    2. Dense retrieval
    3. RRF without reranking
    4. RRF followed by Cross-Encoder reranking

    None evaluates all eligible test queries.
    """
    corpus, queries, qrels, query_ids = load_evaluation_data()

    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive or None.")

        query_ids = query_ids[:max_queries]

    print(f"\nEvaluating {len(query_ids)} queries at k={TOP_K}.")

    # Build baseline retrievers.
    bm25_index, bm25_ids = build_bm25(corpus)
    dense_index, dense_ids = build_dense_index()

    # Build the existing Phase 3 hybrid components.
    (
        hybrid_ids,
        hybrid_bm25_index,
        hybrid_bm25_ids,
        hybrid_faiss_index,
    ) = build_hybrid_index(corpus)

    # Verify document-ID-to-vector-row alignment.
    validate_id_mappings(
        corpus=corpus,
        dense_ids=dense_ids,
        hybrid_ids=hybrid_ids,
        hybrid_bm25_ids=hybrid_bm25_ids,
    )

    # Confirm that both indexes refer to the same corpus.
    if bm25_ids != list(corpus.keys()):
        raise ValueError("Baseline BM25 document mapping is inconsistent.")

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Loading Cross-Encoder...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    method_names = (
        "bm25",
        "dense",
        "rrf",
        "hybrid_reranked",
    )

    all_results = {
        method: []
        for method in method_names
    }

    per_query = []

    for number, query_id in enumerate(query_ids, start=1):
        query = queries[query_id]
        relevant = qrels[query_id]

        print(
            f"\n[{number}/{len(query_ids)}] "
            f"Query ID: {query_id}"
        )

        # ----------------------------------------------------
        # A. BM25 BASELINE
        # ----------------------------------------------------
        bm25_scores = bm25_index.get_scores(tokenize(query))

        bm25_positions = np.argsort(-bm25_scores)[:TOP_K]

        bm25_results = [
            bm25_ids[int(position)]
            for position in bm25_positions
        ]

        # ----------------------------------------------------
        # B. DENSE BASELINE
        # ----------------------------------------------------
        query_vector = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        _, dense_positions = dense_index.search(
            np.ascontiguousarray(query_vector),
            min(TOP_K, dense_index.ntotal),
        )

        dense_results = [
            dense_ids[int(position)]
            for position in dense_positions[0]
            if position >= 0
        ]

        # ----------------------------------------------------
        # C. RRF WITHOUT RERANKING
        # ----------------------------------------------------
        rrf_results = retrieve_rrf_only(
            query=query,
            embedding_model=embedding_model,
            document_ids=hybrid_ids,
            bm25_index=hybrid_bm25_index,
            bm25_ids=hybrid_bm25_ids,
            faiss_index=hybrid_faiss_index,
            top_k=TOP_K,
        )

        # ----------------------------------------------------
        # D. RRF + CROSS-ENCODER RERANKING
        # ----------------------------------------------------
        hybrid_results = retrieve_hybrid(
            query=query,
            corpus=corpus,
            embedding_model=embedding_model,
            cross_encoder=cross_encoder,
            document_ids=hybrid_ids,
            bm25_index=hybrid_bm25_index,
            bm25_ids=hybrid_bm25_ids,
            faiss_index=hybrid_faiss_index,
            top_k=TOP_K,
        )

        reranked_ids = [
            str(item["document_id"])
            for item in hybrid_results
        ]

        retrieved_by_method = {
            "bm25": bm25_results,
            "dense": dense_results,
            "rrf": rrf_results,
            "hybrid_reranked": reranked_ids,
        }

        query_record = {
            "query_id": query_id,
            "query": query,
            "relevant_document_count": len(relevant),
            "methods": {},
        }

        for method, retrieved in retrieved_by_method.items():
            metrics = calculate_metrics(
                retrieved,
                relevant,
                TOP_K,
            )

            all_results[method].append(metrics)

            query_record["methods"][method] = {
                "retrieved_document_ids": retrieved,
                "metrics": metrics,
            }

        per_query.append(query_record)

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================
    summary = {}

    for method, records in all_results.items():
        if not records:
            continue

        metric_names = records[0].keys()

        summary[method] = {
            metric: float(
                np.mean([record[metric] for record in records])
            )
            for metric in metric_names
        }

    report = {
        "dataset": "SciFact",
        "split": "test",
        "evaluated_query_count": len(query_ids),
        "k": TOP_K,
        "methods": list(method_names),
        "metrics_definition": {
            "relevance": "Binary: qrels score greater than zero",
            "recall_denominator": (
                "All relevant document IDs listed for each query "
                "in the loaded qrels."
            ),
            "averaging": "Macro average across evaluated queries",
            "rrf": "Reciprocal Rank Fusion without Cross-Encoder reranking",
            "hybrid_reranked": (
                "RRF candidate fusion followed by Cross-Encoder reranking"
            ),
        },
        "summary": summary,
        "per_query": per_query,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "retrieval_metrics.json"

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================
    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Queries evaluated: {len(query_ids)}")
    print(f"Results saved to: {output_path.resolve()}")

    display_names = {
        "bm25": "BM25",
        "dense": "DENSE",
        "rrf": "RRF (NO RERANKING)",
        "hybrid_reranked": "RRF + CROSS-ENCODER",
    }

    for method, metrics in summary.items():
        print(f"\n{display_names[method]}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n==========================================")

    return report


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25, dense, RRF-only, and RRF + Cross-Encoder retrieval."
        )
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=30,
        help=(
            "Number of queries to evaluate. "
            "Use 30 for a quick test, or 0 for all eligible queries."
        ),
    )

    args = parser.parse_args()

    evaluate(
        max_queries=None if args.max_queries == 0 else args.max_queries
    )
