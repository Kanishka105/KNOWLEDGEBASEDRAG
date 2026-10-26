
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the pretrained sentence-embedding model."""
    return SentenceTransformer(MODEL_NAME)


def encode_texts(model, texts: list[str]):
    """Convert text strings into normalized embedding vectors."""
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
