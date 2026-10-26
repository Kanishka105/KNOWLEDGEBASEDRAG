from app.ai.rag.phase6_document_ingestion.indexing.document_indexer import (
    DocumentIndexer,
)


def test_real_faiss_incremental_save_load_search(tmp_path, monkeypatch):
    import app.ai.rag.phase6_document_ingestion.indexing.document_indexer as module

    monkeypatch.setattr(module, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(module, "METADATA_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(module, "ensure_directories", lambda: None)

    # Use the actual embedding model and FAISS index.
    indexer = DocumentIndexer()

    first_chunks = [
        {
            "chunk_id": "doc1_0",
            "document_id": "doc1",
            "filename": "research1.pdf",
            "page_number": 1,
            "text": "Neural networks learn patterns from training data.",
        }
    ]

    indexer.build(first_chunks)
    indexer.save()

    new_chunks = [
        {
            "chunk_id": "doc2_0",
            "document_id": "doc2",
            "filename": "research2.pdf",
            "page_number": 1,
            "text": "Water conservation reduces water waste.",
        }
    ]

    candidate = indexer.create_incremental_copy(new_chunks)

    # Original index must remain unchanged.
    assert indexer.index.ntotal == 1
    assert len(indexer.chunks) == 1

    # Candidate must contain both documents.
    assert candidate.index.ntotal == 2
    assert len(candidate.chunks) == 2

    candidate.save()

    # Reload using the real FAISS persistence implementation.
    restored = DocumentIndexer()
    restored.load()

    assert restored.index.ntotal == 2
    assert len(restored.chunks) == 2

    results = restored.search(
        "How can we reduce water waste?",
        top_k=2,
    )

    assert len(results) == 2
    assert all("similarity_score" in result for result in results)