
from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase3_hybrid_retrieval.hybrid_retrieval import (
    build_hybrid_index,
    retrieve_hybrid,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_NAME,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


def main():
    corpus, queries, qrels = load_scifact(split="test")

    document_ids, bm25, bm25_ids, faiss_index = build_hybrid_index(corpus)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    if isinstance(queries, dict):
        query_id, query_record = next(iter(queries.items()))
    else:
        query_id, query_record = "first", queries[0]

    if isinstance(query_record, str):
        query_text = query_record
    else:
        query_text = (
            query_record.get("query")
            or query_record.get("claim")
            or query_record.get("text")
        )

    rankings = {}

    for k in [20, 40, 60, 100]:
        results = retrieve_hybrid(
            query=query_text,
            corpus=corpus,
            embedding_model=embedding_model,
            cross_encoder=cross_encoder,
            document_ids=document_ids,
            bm25_index=bm25,
            bm25_ids=bm25_ids,
            faiss_index=faiss_index,
            top_k=5,
            rrf_k=k,
        )

        rankings[k] = [item["document_id"] for item in results]
        print(f"\nk={k}: {rankings[k]}")

    print("\nRanking comparison:")
    for k in [40, 60, 100]:
        print(f"k=20 vs k={k}: {rankings[20] == rankings[k]}")


if __name__ == "__main__":
    main()
