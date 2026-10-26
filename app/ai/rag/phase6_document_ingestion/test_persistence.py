
import json

import faiss
import numpy as np
import pytest

from app.ai.rag.phase6_document_ingestion.indexing.document_indexer import (
    DocumentIndexer,
)


class FakeEmbeddingModel:
    """Fake embeddings so tests need no model download or API call."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            text = text.lower()

            if "municipal" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "water" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def isolated_indexer(tmp_path, monkeypatch):
    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "chunks.json"

    monkeypatch.setattr(
        "app.ai.rag.phase6_document_ingestion.indexing.document_indexer.INDEX_PATH",
        index_path,
    )
    monkeypatch.setattr(
        "app.ai.rag.phase6_document_ingestion.indexing.document_indexer.METADATA_PATH",
        metadata_path,
    )

    indexer = DocumentIndexer.__new__(DocumentIndexer)
    indexer.model = FakeEmbeddingModel()
    indexer.index = None
    indexer.chunks = []

    return indexer, index_path, metadata_path


def test_index_save_reload_and_search(isolated_indexer):
    indexer, index_path, metadata_path = isolated_indexer

    chunks = [
        {
            "text": "Municipal awareness helps citizens report issues.",
            "document_id": "doc-1",
            "filename": "municipal.pdf",
            "page_number": 1,
        },
        {
            "text": "Water supply maintenance is a public service.",
            "document_id": "doc-2",
            "filename": "water.pdf",
            "page_number": 2,
        },
    ]

    indexer.build(chunks)
    indexer.save()

    assert index_path.exists()
    assert metadata_path.exists()

    restored = DocumentIndexer.__new__(DocumentIndexer)
    restored.model = FakeEmbeddingModel()
    restored.index = None
    restored.chunks = []

    restored.load()

    assert restored.index.ntotal == 2
    assert restored.chunks == chunks

    results = restored.search("municipal", top_k=1)

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["similarity_score"] == pytest.approx(1.0)


def test_load_rejects_mismatched_metadata(isolated_indexer):
    _, index_path, metadata_path = isolated_indexer

    index = faiss.IndexFlatIP(3)
    index.add(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(index_path))

    metadata_path.write_text(json.dumps([]), encoding="utf-8")

    restored = DocumentIndexer.__new__(DocumentIndexer)
    restored.model = FakeEmbeddingModel()
    restored.index = None
    restored.chunks = []

    with pytest.raises(ValueError, match="counts do not match"):
        restored.load()


def test_incremental_add_preserves_existing_chunks(isolated_indexer):
    indexer, index_path, metadata_path = isolated_indexer

    first_chunk = {
        "text": "Municipal awareness helps citizens report issues.",
        "document_id": "doc-1",
        "filename": "municipal.pdf",
        "page_number": 1,
    }

    second_chunk = {
        "text": "Water supply maintenance is a public service.",
        "document_id": "doc-2",
        "filename": "water.pdf",
        "page_number": 2,
    }

    # Build the initial index.
    indexer.build([first_chunk])
    assert indexer.index.ntotal == 1

    # Add only the new chunk.
    indexer.add_chunks([second_chunk])

    assert indexer.index.ntotal == 2
    assert len(indexer.chunks) == 2
    assert indexer.chunks == [first_chunk, second_chunk]

    # Confirm that the original chunk remains searchable.
    results = indexer.search("municipal", top_k=1)
    assert results[0]["document_id"] == "doc-1"

    # Save and restore the updated index.
    indexer.save()

    restored = DocumentIndexer.__new__(DocumentIndexer)
    restored.model = FakeEmbeddingModel()
    restored.index = None
    restored.chunks = []
    restored.load()

    assert restored.index.ntotal == 2
    assert restored.chunks == [first_chunk, second_chunk]

    results = restored.search("water", top_k=1)
    assert results[0]["document_id"] == "doc-2"

def test_incremental_copy_embeds_only_new_chunks(isolated_indexer):
    indexer, _, _ = isolated_indexer

    first_chunk = {
        "text": "Municipal awareness helps citizens report issues.",
        "document_id": "doc-1",
        "filename": "municipal.pdf",
        "page_number": 1,
    }

    second_chunk = {
        "text": "Water supply maintenance is a public service.",
        "document_id": "doc-2",
        "filename": "water.pdf",
        "page_number": 1,
    }

    # Track every embedding call and the texts passed to the model.
    original_model = indexer.model

    class CountingEmbeddingModel:
        def __init__(self, model):
            self.model = model
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(list(texts))
            return self.model.encode(texts, **kwargs)

    counting_model = CountingEmbeddingModel(original_model)
    indexer.model = counting_model

    # Build the initial index: the first chunk is embedded once.
    indexer.build([first_chunk])

    assert len(counting_model.calls) == 1
    assert len(counting_model.calls[0]) == 1

    # Record the original index state.
    original_index = indexer.index
    original_vector_count = indexer.index.ntotal
    original_chunks = list(indexer.chunks)

    # Incremental copy should embed only the second chunk.
    candidate = indexer.create_incremental_copy([second_chunk])

    # Exactly one additional embedding call, containing only new text.
    assert len(counting_model.calls) == 2
    assert counting_model.calls[1] == [second_chunk["text"]]

    # The original index and metadata remain unchanged.
    assert indexer.index is original_index
    assert indexer.index.ntotal == original_vector_count
    assert indexer.chunks == original_chunks

    # The candidate contains both chunks and both vectors.
    assert candidate.index.ntotal == 2
    assert candidate.chunks == [first_chunk, second_chunk]

    # The candidate shares the same model wrapper, so its embedding
    # call was recorded by counting_model.
    assert candidate.model is counting_model
