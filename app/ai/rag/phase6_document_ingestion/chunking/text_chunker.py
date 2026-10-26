from app.ai.rag.phase6_document_ingestion.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def chunk_pages(
    pages: list[dict],
    document_id: str,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split extracted page text into overlapping character chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    chunk_id = 0
    step = chunk_size - overlap

    for page in pages:
        text = page["text"].strip()
        page_number = page["page_number"]

        for start in range(0, len(text), step):
            content = text[start:start + chunk_size].strip()

            if not content:
                continue

            chunks.append({
                "chunk_id": f"{document_id}_{chunk_id}",
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "text": content,
            })

            chunk_id += 1

    return chunks
