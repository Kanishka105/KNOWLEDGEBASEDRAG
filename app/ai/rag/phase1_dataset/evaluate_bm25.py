
import json
from pathlib import Path

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase1_dataset.bm25_baseline import build_bm25, search


TOP_K = 5
OUTPUT_DIR = Path("outputs/phase1")


def evaluate():
    corpus, queries, qrels = load_scifact(split="test")
    index, document_ids = build_bm25(corpus)

    precisions = []
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    query_results = []

    for query_id, relevant_docs in qrels.items():
        if query_id not in queries:
            continue

        query_text = queries[query_id].get("text", "")
        results = search(
            index, document_ids, corpus, query_text, top_k=TOP_K
        )

        # SciFact relevance labels: positive scores are relevant.
        relevant_ids = {
            doc_id for doc_id, score in relevant_docs.items()
            if score > 0
        }

        if not relevant_ids:
            continue

        retrieved_ids = [result["document_id"] for result in results]
        retrieved_relevant = [
            doc_id for doc_id in retrieved_ids
            if doc_id in relevant_ids
        ]

        precisions.append(
            len(retrieved_relevant) / TOP_K
        )
        recalls.append(
            len(retrieved_relevant) / len(relevant_ids)
        )

        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        # Binary nDCG@5
        dcg = sum(
            1.0 / __import__("math").log2(rank + 1)
            for rank, doc_id in enumerate(retrieved_ids, start=1)
            if doc_id in relevant_ids
        )
        ideal_hits = min(len(relevant_ids), TOP_K)
        idcg = sum(
            1.0 / __import__("math").log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)

        query_results.append({
            "query_id": query_id,
            "query": query_text,
            "relevant_document_ids": sorted(relevant_ids),
            "retrieved_document_ids": retrieved_ids,
            "relevant_retrieved_count": len(retrieved_relevant),
        })

    if not precisions:
        raise ValueError(
            "No evaluable queries found. Check query IDs and qrels."
        )

    metrics = {
        "split": "test",
        "retrieval_method": "BM25Okapi",
        "top_k": TOP_K,
        "evaluated_queries": len(precisions),
        "precision_at_5": sum(precisions) / len(precisions),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "bm25_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "bm25_query_results.json").write_text(
        json.dumps(query_results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    evaluate()
