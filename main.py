
def main():
    print("Loading SciFact dataset...", flush=True)

    corpus, queries, qrels = load_scifact(split="test")

    print("\nBuilding BM25 index...", flush=True)

    index, document_ids = build_bm25(corpus)

    # Select a query that has relevance judgments.
    matching_query_ids = [
        query_id
        for query_id in queries
        if query_id in qrels
    ]

    if not matching_query_ids:
        raise ValueError("No queries with relevance judgments found.")

    query_id = matching_query_ids[0]
    query_text = queries[query_id]["text"]

    print("\n--- SEARCH QUERY ---")
    print("Query ID:", query_id)
    print("Query:", query_text)

    results = search(
        index=index,
        document_ids=document_ids,
        corpus=corpus,
        query=query_text,
        top_k=5,
    )

    print("\n--- TOP 5 SEARCH RESULTS ---")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print("Document ID:", result["document_id"])
        print("Title:", result["title"])
        print("BM25 score:", round(result["score"], 4))
        print("Text:", result["text"][:300])


if __name__ == "__main__":
    main()
