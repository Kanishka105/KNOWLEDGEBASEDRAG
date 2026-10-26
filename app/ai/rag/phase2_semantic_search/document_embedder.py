
from app.ai.rag.phase1_dataset.text_processing import (
    normalize_document_text,
)
from app.ai.rag.phase2_semantic_search.embedding_model import (
    encode_texts,
)


def embed_corpus(corpus: dict, model):
    """Return document IDs, document texts, and their embeddings."""
    document_ids = list(corpus.keys())

    document_texts = [
        normalize_document_text(corpus[document_id])
        for document_id in document_ids
    ]

    if not document_texts or not any(document_texts):
        raise ValueError("The corpus contains no searchable text.")

    embeddings = encode_texts(model, document_texts)

    return document_ids, document_texts, embeddings
