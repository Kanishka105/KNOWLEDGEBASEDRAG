

from statistics import mean

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase3_hybrid_retrieval.hybrid_retrieval import (
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_NAME,
    build_hybrid_index,
    retrieve_hybrid,
)
from app.ai.rag.phase7_advanced_retrieval.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
)
from app.ai.rag.phase7_advanced_retrieval.evaluation.experiment_runner import (
    save_experiment,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


RRF_VALUES = [20, 40, 60, 100]
MAX_QUERIES = 300  # Smoke-test evaluation; increase to 300 after validation.
TOP_K = 5


def get_query_text(query):
    """Extract query text from a string or SciFact query record."""
    if isinstance(query, str):
        return query.strip()

    if isinstance(query, dict):
        for key in ("query", "claim", "text"):
            value = query.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError(f"Cannot extract query text from: {query!r}")


def get_query_id(query, fallback_id):
    """Read the query ID from a query record when available."""
    if isinstance(query, dict):
        for key in ("_id", "id", "query_id"):
            if key in query:
                return str(query[key])

    return str(fallback_id)


def prepare_queries(queries, qrels):
    """Match queries to queries with relevance judgments."""
    if isinstance(queries, dict):
        records = [
            (str(query_id), query)
            for query_id, query in queries.items()
        ]
    else:
        records = [
            (get_query_id(query, index), query)
            for index, query in enumerate(queries)
        ]

    matched = []

    for query_id, query in records:
        if query_id in qrels and qrels[query_id]:
            matched.append((query_id, query))

    if not matched:
        raise ValueError(
            "No queries matched the relevance judgments. "
            "Check query IDs and qrels format."
        )

    return matched[:MAX_QUERIES]


def run_experiments():
    print("Loading SciFact test data...")
    corpus, queries, qrels = load_scifact(split="test")

    matched_queries = prepare_queries(queries, qrels)
    print(f"Queries selected: {len(matched_queries)}")

    print("Building BM25 and FAISS indexes...")
    document_ids, bm25_index, bm25_ids, faiss_index = (
        build_hybrid_index(corpus)
    )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Loading CrossEncoder...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    for rrf_k in RRF_VALUES:
        print(f"\n{'=' * 60}")
        print(f"Evaluating RRF k={rrf_k}")
        print(f"{'=' * 60}")

        precision_values = []
        recall_values = []
        reciprocal_rank_values = []
        ndcg_values = []

        for index, (query_id, query_record) in enumerate(
            matched_queries, start=1
        ):
            query_text = get_query_text(query_record)

            relevant_ids = {
                str(doc_id)
                for doc_id, score in qrels[query_id].items()
                if float(score) > 0
            }

            if not relevant_ids:
                continue

            results = retrieve_hybrid(
                query=query_text,
                corpus=corpus,
                embedding_model=embedding_model,
                cross_encoder=cross_encoder,
                document_ids=document_ids,
                bm25_index=bm25_index,
                bm25_ids=bm25_ids,
                faiss_index=faiss_index,
                top_k=TOP_K,
                rrf_k=rrf_k,
            )

            retrieved_ids = [
                str(result["document_id"])
                for result in results
            ]

            precision_values.append(
                precision_at_k(retrieved_ids, relevant_ids, TOP_K)
            )
            recall_values.append(
                recall_at_k(retrieved_ids, relevant_ids, TOP_K)
            )
            reciprocal_rank_values.append(
                reciprocal_rank(retrieved_ids, relevant_ids)
            )
            ndcg_values.append(
                ndcg_at_k(retrieved_ids, relevant_ids, TOP_K)
            )

            if index % 5 == 0 or index == len(matched_queries):
                print(
                    f"Progress: {index}/{len(matched_queries)} queries"
                )

        if not precision_values:
            raise ValueError(
                "No queries were evaluated. Check query IDs and qrels."
            )

        metrics = {
            "precision@5": mean(precision_values),
            "recall@5": mean(recall_values),
            "mrr@5": mean(reciprocal_rank_values),
            "ndcg@5": mean(ndcg_values),
        }

        print(f"\nResults for RRF k={rrf_k}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        save_experiment(
            name=f"hybrid_rrf_k{rrf_k}_test",
            method="BM25 + FAISS + RRF + CrossEncoder",
            parameters={
                "rrf_k": rrf_k,
                "top_k": TOP_K,
                "candidate_k": 100,
                "rerank_k": 50,
                "evaluated_queries": len(precision_values),
                "split": "test",
            },
            metrics=metrics,
            notes=(
                "Hybrid retrieval evaluated on the same SciFact test "
                "queries with binary relevance and fixed retrieval settings."
            ),
        )

    print("\nAll RRF experiments completed.")
    print("Results saved to reports/experiment_results.json")


if __name__ == "__main__":
    run_experiments()
