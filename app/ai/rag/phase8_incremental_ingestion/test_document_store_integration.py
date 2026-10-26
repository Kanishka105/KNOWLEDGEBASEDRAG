
import hashlib
import json

import pytest

from app.ai.rag.phase6_document_ingestion import (
    document_store as store_module,
)


class FakeIndex:
    """Minimal FAISS-like index for integration tests."""

    def __init__(self, count=0):
        self.ntotal = count


class FakeIndexer:
    """Lightweight indexer for testing DocumentStore integration."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.added_batches = []
        self.save_count = 0

    def build(self, chunks):
        self.chunks = list(chunks)
        self.index = FakeIndex(len(self.chunks))

    def add_chunks(self, chunks):
        if not chunks:
            return

        self.added_batches.append(list(chunks))
        self.chunks.extend(chunks)

        if self.index is None:
            self.index = FakeIndex()

        self.index.ntotal += len(chunks)

    def create_incremental_copy(self, new_chunks):
        """Copy existing chunks and append only new chunks."""
        candidate = FakeIndexer()

        candidate.chunks = list(self.chunks)
        candidate.index = FakeIndex(
            self.index.ntotal if self.index is not None else 0
        )
        candidate.add_chunks(new_chunks)

        return candidate

    def save(self):
        """Persist fake chunk metadata for the test."""
        self.save_count += 1

        store_module.METADATA_PATH.write_text(
            json.dumps(
                self.chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self):
        """Restore fake index metadata from disk."""
        self.chunks = json.loads(
            store_module.METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )
        self.index = FakeIndex(len(self.chunks))

    def search(self, query, top_k=5):
        return self.chunks[:top_k]


@pytest.fixture
def ingestion_setup(tmp_path, monkeypatch):
    """Set up an isolated DocumentStore without real PDFs or embeddings."""
    documents_path = tmp_path / "documents.json"
    metadata_path = tmp_path / "chunks.json"

    monkeypatch.setattr(
        store_module,
        "DOCUMENTS_PATH",
        documents_path,
    )
    monkeypatch.setattr(
        store_module,
        "METADATA_PATH",
        metadata_path,
    )
    monkeypatch.setattr(
        store_module,
        "ensure_directories",
        lambda: None,
    )
    monkeypatch.setattr(
        store_module,
        "DocumentIndexer",
        FakeIndexer,
    )

    def fake_extract_pdf_pages(path):
        return [
            {
                "page_number": 1,
                "text": f"Text from {path.name}",
            }
        ]

    def fake_chunk_pages(pages, document_id, filename):
        return [
            {
                "text": pages[0]["text"],
                "document_id": document_id,
                "filename": filename,
                "page_number": pages[0]["page_number"],
            }
        ]

    monkeypatch.setattr(
        store_module,
        "extract_pdf_pages",
        fake_extract_pdf_pages,
    )
    monkeypatch.setattr(
        store_module,
        "chunk_pages",
        fake_chunk_pages,
    )

    store = store_module.DocumentStore()

    return store, documents_path, metadata_path


def test_duplicate_pdf_is_not_ingested_twice(
    ingestion_setup, tmp_path
):
    store, documents_path, metadata_path = ingestion_setup

    pdf_path = tmp_path / "municipal.pdf"
    pdf_path.write_bytes(b"same PDF content")

    first_result = store.ingest_pdf(pdf_path)
    original_chunk_count = len(store.chunks)
    original_save_count = store.indexer.save_count

    # Different filename, but identical file content.
    duplicate_path = tmp_path / "renamed_municipal.pdf"
    duplicate_path.write_bytes(b"same PDF content")

    second_result = store.ingest_pdf(duplicate_path)

    assert first_result["status"] == "ingested"
    assert second_result["status"] == "already_ingested"

    assert len(store.documents) == 1
    assert len(store.chunks) == original_chunk_count
    assert store.indexer.save_count == original_save_count

    saved_documents = json.loads(
        documents_path.read_text(encoding="utf-8")
    )
    saved_chunks = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert len(saved_documents) == 1
    assert len(saved_chunks) == 1


def test_new_pdf_appends_chunks_and_preserves_existing(
    ingestion_setup, tmp_path
):
    store, documents_path, metadata_path = ingestion_setup

    first_pdf = tmp_path / "municipal.pdf"
    first_pdf.write_bytes(b"municipal PDF content")

    store.ingest_pdf(first_pdf)

    original_chunk = dict(store.chunks[0])

    second_pdf = tmp_path / "water.pdf"
    second_pdf.write_bytes(b"water PDF content")

    result = store.ingest_pdf(second_pdf)

    assert result["status"] == "ingested"
    assert len(store.documents) == 2
    assert len(store.chunks) == 2

    # Existing metadata remains unchanged.
    assert store.chunks[0] == original_chunk
    assert store.chunks[1]["filename"] == "water.pdf"

    # Candidate index contains old and new chunks.
    assert len(store.indexer.chunks) == 2
    assert store.indexer.chunks[0] == original_chunk
    assert store.indexer.chunks[1]["filename"] == "water.pdf"
    assert store.indexer.index.ntotal == 2

    saved_documents = json.loads(
        documents_path.read_text(encoding="utf-8")
    )
    saved_chunks = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert len(saved_documents) == 2
    assert len(saved_chunks) == 2
    assert saved_chunks[0] == original_chunk
    assert saved_chunks[1]["filename"] == "water.pdf"

    expected_id = hashlib.sha256(
        b"water PDF content"
    ).hexdigest()[:16]

    assert result["document_id"] == expected_id


def test_document_store_reloads_persisted_state(
    ingestion_setup, tmp_path
):
    store, documents_path, metadata_path = ingestion_setup

    # Ingest two different PDFs.
    first_pdf = tmp_path / "municipal.pdf"
    first_pdf.write_bytes(b"municipal document content")
    store.ingest_pdf(first_pdf)

    second_pdf = tmp_path / "water.pdf"
    second_pdf.write_bytes(b"water document content")
    store.ingest_pdf(second_pdf)

    expected_documents = list(store.documents)
    expected_chunks = list(store.chunks)

    assert documents_path.exists()
    assert metadata_path.exists()

    # Simulate a restart by constructing a new DocumentStore.
    restarted_store = store_module.DocumentStore()

    assert restarted_store.documents == expected_documents
    assert restarted_store.chunks == expected_chunks

    assert restarted_store.indexer.index.ntotal == len(expected_chunks)
    assert restarted_store.indexer.chunks == expected_chunks

def test_failed_save_preserves_active_in_memory_state(
    ingestion_setup, tmp_path, monkeypatch
):
    store, _, _ = ingestion_setup

    # Ingest the first PDF successfully.
    first_pdf = tmp_path / "municipal.pdf"
    first_pdf.write_bytes(b"municipal document content")
    store.ingest_pdf(first_pdf)

    # Keep references to the active state.
    original_indexer = store.indexer
    original_chunks = list(store.chunks)
    original_documents = list(store.documents)

    # Prepare a different PDF.
    second_pdf = tmp_path / "water.pdf"
    second_pdf.write_bytes(b"water document content")

    # Simulate a failure while saving the candidate index.
    def fail_save(self):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(FakeIndexer, "save", fail_save)

    # Ingestion should fail, but active in-memory state must remain intact.
    with pytest.raises(OSError, match="Simulated disk write failure"):
        store.ingest_pdf(second_pdf)

    assert store.indexer is original_indexer
    assert store.chunks == original_chunks
    assert store.documents == original_documents
def test_failed_document_registry_write_preserves_active_state(
    ingestion_setup, tmp_path, monkeypatch
):
    store, documents_path, metadata_path = ingestion_setup

    first_pdf = tmp_path / "municipal.pdf"
    first_pdf.write_bytes(b"municipal document content")
    store.ingest_pdf(first_pdf)

    original_indexer = store.indexer
    original_chunks = list(store.chunks)
    original_documents = list(store.documents)

    second_pdf = tmp_path / "water.pdf"
    second_pdf.write_bytes(b"water document content")

    def fail_registry_write(path, data):
        if path == store_module.DOCUMENTS_PATH:
            raise OSError("Simulated registry write failure")

        # Preserve normal behavior for other JSON writes.
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    monkeypatch.setattr(
        store_module.DocumentStore,
        "_write_json",
        staticmethod(fail_registry_write),
    )

    with pytest.raises(OSError, match="Simulated registry write failure"):
        store.ingest_pdf(second_pdf)

    # The active in-memory state must remain unchanged.
    assert store.indexer is original_indexer
    assert store.chunks == original_chunks
    assert store.documents == original_documents