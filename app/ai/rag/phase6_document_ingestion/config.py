from pathlib import Path

# Project root: knowledgeBasedRag/
PROJECT_ROOT = Path(__file__).resolve().parents[4]

UPLOAD_DIR = PROJECT_ROOT / "datasets" / "uploaded_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase6"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Reuse the same embedding model as Phase 2/3.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Saved index and metadata
INDEX_PATH = OUTPUT_DIR / "uploaded_documents.faiss"
METADATA_PATH = OUTPUT_DIR / "chunks_metadata.json"
DOCUMENTS_PATH = OUTPUT_DIR / "documents.json"


def ensure_directories() -> None:
    """Create storage directories if they do not exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
