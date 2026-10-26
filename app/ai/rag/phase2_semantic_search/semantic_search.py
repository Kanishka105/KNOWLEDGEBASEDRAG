
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

STORE_DIR = Path("outputs/phase2/vector_store")


def main():
    print("Loading SciFact dataset...")
    corpus, queries, qrels = load_scifact(split="test")

    model = load_embedding_model()

    if (
        (STORE_DIR / "embeddings.npy").exists()
        and (STORE_DIR / "document_ids.json").exists()
    ):
        print("Loading saved document embeddings...")
        document_ids, embeddings = load_vector_store(STORE_DIR)

        # Prevent using vectors with an incompatible corpus ordering.
        if document_ids != list(corpus.keys()):
            raise ValueError(
                "Saved embeddings do not match the current corpus order. "
                "Rebuild the vector store."
            )
    else:
        print("Creating document embeddings...")
        document_ids, document_texts, embeddings = embed_corpus(
            corpus, model
        )
        save_vector_store(STORE_DIR, document_ids, embeddings)

    matching_query_ids = [
        query_id for query_id in queries if query_id in qrels
    ]

    if not matching_query_ids:
        raise ValueError("No queries with relevance judgments found.")

    query_id = matching_query_ids[0]
    query_text = queries[query_id].get("text", "")

    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query_text}")

    query_embedding = encode_texts(model, [query_text])[0]

    results = search_vectors(
        query_embedding,
        document_ids,
        embeddings,
        top_k=5,
    )

    print("\nTop 5 semantic-search results:")

    for result in results:
        document = corpus[result["document_id"]]

        print(f"\nRank: {result['rank']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Title: {document.get('title', 'No title')}")
        print(f"Cosine similarity: {result['score']:.4f}")


if __name__ == "__main__":
    main()
