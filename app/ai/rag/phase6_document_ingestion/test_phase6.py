
def test_duplicate_pdf_is_not_ingested_twice(tmp_path, monkeypatch):
    """Same PDF content should be detected by SHA-256."""

    import hashlib

    from app.ai.rag.phase6_document_ingestion.document_store import (
        DocumentStore,
    )

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake test content")

    digest = hashlib.sha256(pdf_file.read_bytes()).hexdigest()

    store = DocumentStore.__new__(DocumentStore)
    store.documents = [
        {
            "document_id": digest[:16],
            "filename": "sample.pdf",
            "sha256": digest,
            "page_count": 1,
            "chunk_count": 2,
        }
    ]

    result = store.ingest_pdf(pdf_file)

    assert result["status"] == "already_ingested"
    assert result["document_id"] == digest[:16]
    assert result["chunk_count"] == 2
