
from rank_bm25 import BM25Okapi

from app.ai.rag.phase1_dataset.dataset_loader import load_scifact
from app.ai.rag.phase1_dataset.text_processing import (
    normalize_document_text,
    tokenize,
)


def build_bm25(corpus: dict):
    document_ids = list(corpus.keys())
    tokenized_documents = []

    for document_id in document_ids:
        text = normalize_document_text(corpus[document_id])
        tokenized_documents.append(tokenize(text))

    if not tokenized_documents or not any(tokenized_documents):
        raise ValueError("The corpus contains no searchable text.")

    index = BM25Okapi(tokenized_documents)
    return index, document_ids


def search(
    index,
    document_ids: list[str],
    corpus: dict,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    ranked_positions = sorted(
        range(len(scores)),
        key=lambda position: scores[position],
        reverse=True,
    )[:top_k]

    results = []

    for rank, position in enumerate(ranked_positions, start=1):
        document_id = document_ids[position]
        document = corpus[document_id]

        results.append({
            "rank": rank,
            "document_id": document_id,
            "title": document.get("title", "No title"),
            "score": float(scores[position]),
            "text": normalize_document_text(document),
        })

    return results


def main():
    print("Loading SciFact...", flush=True)
    corpus, queries, qrels = load_scifact(split="test")

    print("Building BM25 index...", flush=True)
    index, document_ids = build_bm25(corpus)

    matching_query_ids = [
        query_id for query_id in queries
        if query_id in qrels
    ]

    if not matching_query_ids:
        raise ValueError(
            "No query IDs match between queries and relevance judgments."
        )

    query_id = matching_query_ids[0]
    query_text = queries[query_id].get("text", "")

    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query_text}")

    results = search(
        index, document_ids, corpus, query_text, top_k=5
    )

    print("\nTop 5 retrieved papers:")
    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Title: {result['title']}")
        print(f"BM25 score: {result['score']:.4f}")


if __name__ == "__main__":
    main()
