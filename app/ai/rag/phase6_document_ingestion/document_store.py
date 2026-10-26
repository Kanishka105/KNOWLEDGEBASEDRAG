
import hashlib
import json
from pathlib import Path

from app.ai.rag.phase6_document_ingestion.config import (
    DOCUMENTS_PATH,
    METADATA_PATH,
    ensure_directories,
)
from app.ai.rag.phase6_document_ingestion.parsers.pdf_parser import (
    extract_pdf_pages,
)
from app.ai.rag.phase6_document_ingestion.chunking.text_chunker import (
    chunk_pages,
)
from app.ai.rag.phase6_document_ingestion.indexing.document_indexer import (
    DocumentIndexer,
)


class DocumentStore:
    """Manage PDF ingestion, deduplication, indexing, and retrieval."""

    def __init__(self):
        ensure_directories()

        self.documents = self._load_json(DOCUMENTS_PATH, [])
        self.chunks = self._load_json(METADATA_PATH, [])

        self.indexer = DocumentIndexer()

        # Load the persisted index if chunks already exist.
        if self.chunks:
            self.indexer.load()

            if self.indexer.chunks != self.chunks:
                raise ValueError(
                    "DocumentStore metadata and indexer metadata differ."
                )

    @staticmethod
    def _load_json(path: Path, default):
        """Load JSON data or return the supplied default."""
        if not path.exists():
            return default

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data) -> None:
        """Write JSON data to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def ingest_pdf(self, pdf_path: str | Path) -> dict:
        """Ingest a PDF using incremental indexing.

        Duplicate PDFs are skipped using their SHA-256 hash.
        Existing vectors are copied into a candidate index, and only
        newly added chunks are embedded when an index already exists.
        The active in-memory index is replaced after persistence succeeds.
        """
        path = Path(pdf_path)

        # 1. Validate the PDF path.
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        # 2. Calculate a content-based hash.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        # 3. Skip PDFs whose content has already been ingested.
        for document in self.documents:
            if document.get("sha256") == digest:
                return {
                    "status": "already_ingested",
                    "document_id": document["document_id"],
                    "chunk_count": document["chunk_count"],
                }

        # 4. Extract text from the PDF.
        pages = extract_pdf_pages(path)

        document_id = digest[:16]

        # 5. Split extracted pages into chunks.
        new_chunks = chunk_pages(
            pages=pages,
            document_id=document_id,
            filename=path.name,
        )

        if not new_chunks:
            raise ValueError(
                "No text chunks were extracted from this PDF."
            )

        # 6. Prepare the document registry entry.
        document = {
            "document_id": document_id,
            "filename": path.name,
            "sha256": digest,
            "page_count": len(pages),
            "chunk_count": len(new_chunks),
        }

        updated_chunks = self.chunks + new_chunks
        updated_documents = self.documents + [document]

        # 7. Create an independent candidate index.
        # If an index exists, preserve its vectors and embed only
        # the newly extracted chunks.
        if self.indexer.index is not None:
            candidate_indexer = (
                self.indexer.create_incremental_copy(new_chunks)
            )
        else:
            # First ingestion: build an index from the new chunks.
            candidate_indexer = DocumentIndexer()
            candidate_indexer.build(new_chunks)

        # 8. Verify that the candidate contains all expected chunks.
        if candidate_indexer.chunks != updated_chunks:
            raise ValueError(
                "Candidate index chunks do not match the updated "
                "document metadata."
            )

        if (
            candidate_indexer.index is None
            or candidate_indexer.index.ntotal != len(updated_chunks)
        ):
            raise ValueError(
                "Candidate index vector count does not match "
                "the expected chunk count."
            )

        # 9. Persist the candidate index and its matching chunk metadata.
        candidate_indexer.save()

        # 10. Persist the updated document registry.
        self._write_json(
            DOCUMENTS_PATH,
            updated_documents,
        )

        # 11. Replace the active in-memory state.
        self.indexer = candidate_indexer
        self.chunks = updated_chunks
        self.documents = updated_documents

        return {
            "status": "ingested",
            **document,
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search indexed document chunks."""
        if self.indexer.index is None:
            raise ValueError(
                "No indexed documents are available."
            )

        return self.indexer.search(
            query,
            top_k=top_k,
        )
