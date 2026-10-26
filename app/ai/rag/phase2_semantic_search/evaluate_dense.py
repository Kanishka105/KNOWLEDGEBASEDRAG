
import json
import math
from pathlib import Path

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase2_semantic_search.embedding_model import (
    load_embedding_model,
    encode_texts,
)
from app.ai.rag.phase2_semantic_search.document_embedder import (
    embed_corpus,
)
from app.ai.rag.phase2_semantic_search.vector_store import (
    save_vector_store,
    load_vector_store,
    search_vectors,
)

TOP_K = 5
STORE_DIR = Path("outputs/phase2/vector_store")
RESULT_DIR = Path("outputs/phase2")


def get_vector_store(corpus, model):
    ids_path = STORE_DIR / "document_ids.json"
    embeddings_path = STORE_DIR / "embeddings.npy"

    if ids_path.exists() and embeddings_path.exists():
        document_ids, embeddings = load_vector_store(STORE_DIR)

        if document_ids == list(corpus.keys()):
            return document_ids, embeddings

        print("Corpus order changed. Rebuilding embeddings...")

    document_ids, _, embeddings = embed_corpus(corpus, model)
    save_vector_store(STORE_DIR, document_ids, embeddings)

    return document_ids, embeddings


def evaluate():
    corpus, queries, qrels = load_scifact(split="test")
    model = load_embedding_model()
    document_ids, embeddings = get_vector_store(corpus, model)

    precisions = []
    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    query_results = []

    for query_id, judgments in qrels.items():
        if query_id not in queries:
            continue

        relevant_ids = {
            doc_id for doc_id, score in judgments.items()
            if score > 0
        }

        if not relevant_ids:
            continue

        query_text = queries[query_id].get("text", "")
        query_embedding = encode_texts(model, [query_text])[0]

        results = search_vectors(
            query_embedding,
            document_ids,
            embeddings,
            top_k=TOP_K,
        )

        retrieved_ids = [item["document_id"] for item in results]
        hits = [
            doc_id for doc_id in retrieved_ids
            if doc_id in relevant_ids
        ]

        precisions.append(len(hits) / TOP_K)
        recalls.append(len(hits) / len(relevant_ids))

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(retrieved_ids, start=1)
            if doc_id in relevant_ids
        )

        ideal_hits = min(len(relevant_ids), TOP_K)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)

        query_results.append({
            "query_id": query_id,
            "query": query_text,
            "relevant_document_ids": sorted(relevant_ids),
            "retrieved_document_ids": retrieved_ids,
            "relevant_retrieved_count": len(hits),
        })

    if not precisions:
        raise ValueError("No evaluable queries found.")

    metrics = {
        "split": "test",
        "retrieval_method": "Dense cosine similarity",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": TOP_K,
        "evaluated_queries": len(precisions),
        "precision_at_5": sum(precisions) / len(precisions),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr_at_5": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "ndcg_at_5": sum(ndcgs) / len(ndcgs),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    (RESULT_DIR / "dense_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (RESULT_DIR / "dense_query_results.json").write_text(
        json.dumps(query_results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved in: {RESULT_DIR.resolve()}")


if __name__ == "__main__":
    evaluate()
