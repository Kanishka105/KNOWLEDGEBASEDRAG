
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return text.lower().split()


def build_bm25(corpus: dict):
    """Build the keyword-based BM25 index."""
    document_ids = list(corpus.keys())
    tokenized_documents = []

    for document_id in document_ids:
        document = corpus[document_id]
        title = document.get("title", "")
        text = document.get("text", "")

        tokenized_documents.append(
            tokenize(f"{title} {text}")
        )

    index = BM25Okapi(tokenized_documents)

    return index, document_ids


def build_vector_index(corpus: dict, model):
    """Encode documents and build a FAISS vector index."""
    document_ids = list(corpus.keys())
    documents = []

    for document_id in document_ids:
        document = corpus[document_id]
        title = document.get("title", "")
        text = document.get("text", "")

        documents.append(f"{title}. {text}")

    print("Encoding documents for semantic search...", flush=True)

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, document_ids


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked document IDs using RRF."""
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def hybrid_search(
    query: str,
    corpus: dict,
    bm25_index,
    bm25_document_ids: list[str],
    embedding_model,
    vector_index,
    vector_document_ids: list[str],
    reranker,
    top_k: int = 5,
    candidate_k: int = 50,
) -> list[dict]:
    """Combine BM25 and vector retrieval, then rerank candidates."""

    if top_k < 1 or candidate_k < 1:
        raise ValueError("top_k and candidate_k must be at least 1")

    # 1. Retrieve candidates using BM25.
    bm25_scores = bm25_index.get_scores(tokenize(query))

    bm25_positions = sorted(
        range(len(bm25_scores)),
        key=lambda position: bm25_scores[position],
        reverse=True,
    )[:candidate_k]

    bm25_ranking = [
        bm25_document_ids[position]
        for position in bm25_positions
    ]

    # 2. Retrieve candidates using semantic similarity.
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    semantic_k = min(candidate_k, vector_index.ntotal)
    _, vector_positions = vector_index.search(
        query_embedding,
        semantic_k,
    )

    semantic_ranking = [
        vector_document_ids[int(position)]
        for position in vector_positions[0]
        if position >= 0
    ]

    # 3. Fuse the two rankings.
    fused_results = reciprocal_rank_fusion(
        [bm25_ranking, semantic_ranking]
    )

    candidate_ids = [
        document_id
        for document_id, _ in fused_results[:candidate_k]
    ]

    # 4. Rerank candidates using query-document relevance.
    pairs = [
        [
            query,
            (
                f"{corpus[document_id].get('title', '')}. "
                f"{corpus[document_id].get('text', '')}"
            ),
        ]
        for document_id in candidate_ids
    ]

    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=True,
    )

    reranked = sorted(
        zip(candidate_ids, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    results = []

    for rank, (document_id, score) in enumerate(
        reranked,
        start=1,
    ):
        document = corpus[document_id]

        results.append({
            "rank": rank,
            "document_id": document_id,
            "title": document.get("title", ""),
            "rerank_score": float(score),
            "text": document.get("text", ""),
        })

    return results


def main():
    print("Loading SciFact...", flush=True)
    corpus, queries, qrels = load_scifact(split="test")

    print("\nLoading embedding model...", flush=True)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nBuilding BM25 index...", flush=True)
    bm25_index, bm25_document_ids = build_bm25(corpus)

    print("\nBuilding FAISS index...", flush=True)
    vector_index, vector_document_ids = build_vector_index(
        corpus,
        embedding_model,
    )

    print("\nLoading reranker...", flush=True)
    reranker = CrossEncoder(RERANKER_MODEL)

    matching_query_ids = [
        query_id
        for query_id in queries
        if query_id in qrels
    ]

    if not matching_query_ids:
        raise ValueError("No queries with relevance judgments found.")

    query_id = matching_query_ids[0]
    query_text = queries[query_id]["text"]

    print("\n--- HYBRID SEARCH QUERY ---")
    print("Query ID:", query_id)
    print("Query:", query_text)

    results = hybrid_search(
        query=query_text,
        corpus=corpus,
        bm25_index=bm25_index,
        bm25_document_ids=bm25_document_ids,
        embedding_model=embedding_model,
        vector_index=vector_index,
        vector_document_ids=vector_document_ids,
        reranker=reranker,
        top_k=5,
        candidate_k=50,
    )

    print("\n--- TOP 5 HYBRID RESULTS ---")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print("Document ID:", result["document_id"])
        print("Title:", result["title"])
        print("Rerank score:", round(result["rerank_score"], 4))
        print("Text:", result["text"][:300])


if __name__ == "__main__":
    main()
