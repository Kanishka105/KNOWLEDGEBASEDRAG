
from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase2_semantic_search.semantic_search import (
    load_embedding_model,
    build_vector_index,
    search,
)
from app.ai.rag.phase5_evaluation.evaluate_retrieval import (
    recall_at_k,
    reciprocal_rank,
    average_precision_at_k,
)


def main():
    print("Loading SciFact...", flush=True)
    corpus, queries, qrels = load_scifact(split="test")

    model = load_embedding_model()
    index, document_ids = build_vector_index(corpus, model)

    recall_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    evaluated_queries = 0

    matching_ids = [
        query_id for query_id in queries
        if query_id in qrels
    ]

    for query_id in matching_ids:
        relevant_ids = {
            document_id
            for document_id, score in qrels[query_id].items()
            if score > 0 and document_id in corpus
        }

        if not relevant_ids:
            continue

        results = search(
            query=queries[query_id]["text"],
            model=model,
            index=index,
            document_ids=document_ids,
            corpus=corpus,
            top_k=5,
        )

        retrieved_ids = [
            result["document_id"] for result in results
        ]

        recall_total += recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        mrr_total += reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        map_total += average_precision_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        evaluated_queries += 1

    if evaluated_queries == 0:
        raise ValueError("No queries available for evaluation.")

    print("\n--- SEMANTIC SEARCH EVALUATION ---")
    print("Queries evaluated:", evaluated_queries)
    print("Recall@5:", round(recall_total / evaluated_queries, 4))
    print("MRR@5:", round(mrr_total / evaluated_queries, 4))
    print("MAP@5:", round(map_total / evaluated_queries, 4))


if __name__ == "__main__":
    main()
