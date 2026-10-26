
from app.ai.rag.phase1_dataset.dataset_loader import load_scifact


def main():
    corpus, queries, qrels = load_scifact(split="test")

    print("\n--- SCIFACT DATASET SUMMARY ---")
    print("Documents:", len(corpus))
    print("Queries:", len(queries))
    print("Queries with relevance judgments:", len(qrels))

    # Select a query that has relevance judgments.
    matching_query_ids = [
        query_id for query_id in qrels
        if query_id in queries
    ]

    print(
        "Queries available for evaluation:",
        len(matching_query_ids),
    )

    if not matching_query_ids:
        print("No matching query IDs found.")
        return

    query_id = matching_query_ids[0]
    query = queries[query_id]

    print("\n--- SAMPLE QUERY ---")
    print("ID:", query_id)
    print("Text:", query.get("text", ""))

    print("\n--- RELEVANT DOCUMENTS ---")
    relevant_documents = qrels[query_id]

    for document_id, score in relevant_documents.items():
        document = corpus.get(document_id)

        print("\nDocument ID:", document_id)
        print("Relevance score:", score)

        if document:
            print("Title:", document.get("title", "No title"))
        else:
            print("Document not found in corpus.")

    print("\n--- SAMPLE CORPUS DOCUMENT ---")
    document_id = next(iter(corpus))
    document = corpus[document_id]

    print("ID:", document_id)
    print("Title:", document.get("title", "No title"))
    print("Text:", document.get("text", "")[:300])


if __name__ == "__main__":
    main()
