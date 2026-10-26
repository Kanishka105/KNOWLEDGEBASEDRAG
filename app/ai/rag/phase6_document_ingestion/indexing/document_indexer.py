
import json

import faiss
from sentence_transformers import SentenceTransformer

from app.ai.rag.phase6_document_ingestion.config import (
    EMBEDDING_MODEL_NAME,
    INDEX_PATH,
    METADATA_PATH,
    ensure_directories,
)


class DocumentIndexer:
    """Build, incrementally update, save, load, and search a FAISS index."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        """Build a new index from all supplied chunks."""
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")

        texts = [chunk["text"] for chunk in chunks]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = list(chunks)

    def add_chunks(self, new_chunks: list[dict]) -> None:
        """Embed and append only new chunks to the existing FAISS index."""
        if not new_chunks:
            return

        if self.index is None:
            self.build(new_chunks)
            return

        texts = [chunk["text"] for chunk in new_chunks]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(
                "New embedding dimensions do not match the existing index."
            )

        if len(vectors) != len(new_chunks):
            raise ValueError(
                "The number of embeddings does not match the number of chunks."
            )

        # Append vectors and metadata in the same order.
        self.index.add(vectors)
        self.chunks.extend(new_chunks)

    def save(self) -> None:
        """Persist the FAISS index and its corresponding chunk metadata."""
        if self.index is None:
            raise ValueError("Build the index before saving it.")

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and metadata counts do not match."
            )

        ensure_directories()

        faiss.write_index(self.index, str(INDEX_PATH))

        with METADATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self) -> None:
        """Load the persisted index and verify metadata alignment."""
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Index or metadata is missing. Build the index first."
            )

        index = faiss.read_index(str(INDEX_PATH))

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS index and metadata counts do not match."
            )

        self.index = index
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index using cosine similarity of normalized vectors."""
        if self.index is None:
            raise ValueError("Build or load the index before searching.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimensions do not match the index."
            )

        scores, positions = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue

            result = dict(self.chunks[int(position)])
            result["similarity_score"] = float(score)
            results.append(result)

        return results

    def create_incremental_copy(self, new_chunks: list[dict]):
        """Create an independent index containing old and new chunks.

        Only the new chunks are embedded. The current index remains
        unchanged while the candidate index is prepared.
        """
        if self.index is None:
            candidate = DocumentIndexer.__new__(DocumentIndexer)
            candidate.model = self.model
            candidate.index = None
            candidate.chunks = []
            candidate.build(new_chunks)
            return candidate

        if not new_chunks:
            raise ValueError("Provide at least one new chunk.")

        candidate = DocumentIndexer.__new__(DocumentIndexer)
        candidate.model = self.model

        # Copy the existing FAISS index so the active index isn't mutated.
        candidate.index = faiss.clone_index(self.index)
        candidate.chunks = list(self.chunks)

        # Embed and append only the new chunks to the independent copy.
        candidate.add_chunks(new_chunks)

        return candidate

