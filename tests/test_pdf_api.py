
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Use a temporary upload directory for every test.
    monkeypatch.setattr(api_main, "UPLOAD_DIR", tmp_path)

    with TestClient(api_main.app) as test_client:
        # Override lifespan initialization so tests don't load real models.
        api_main.app.state.rag_ready = False
        api_main.app.state.documents_ready = True
        api_main.app.state.gemini_client = Mock()
        api_main.app.state.rag_components = None
        api_main.app.state.document_store = Mock()

        yield test_client

    # Clean up test state.
    for name in (
        "rag_ready",
        "documents_ready",
        "gemini_client",
        "rag_components",
        "document_store",
    ):
        if hasattr(api_main.app.state, name):
            delattr(api_main.app.state, name)


def test_list_documents_returns_documents(client):
    store = api_main.app.state.document_store
    store.documents = [
        {
            "document_id": "doc-001",
            "filename": "report.pdf",
            "chunk_count": 3,
        }
    ]

    response = client.get("/api/v1/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["documents"][0]["document_id"] == "doc-001"


def test_search_documents_returns_results(client):
    store = api_main.app.state.document_store
    store.search.return_value = [
        {
            "document_id": "doc-001",
            "filename": "report.pdf",
            "page_number": 2,
            "text": "The report discusses image authenticity.",
            "similarity_score": 0.91,
        }
    ]

    response = client.post(
        "/api/v1/documents/search",
        json={"query": "image authenticity", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["filename"] == "report.pdf"
    assert data["results"][0]["page_number"] == 2

    store.search.assert_called_once_with(
        query="image authenticity",
        top_k=3,
    )


def test_search_rejects_too_short_query(client):
    response = client.post(
        "/api/v1/documents/search",
        json={"query": "AI"},
    )

    assert response.status_code == 422


def test_upload_rejects_non_pdf_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "pdf extension" in response.json()["detail"].lower()


def test_upload_rejects_invalid_pdf_signature(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "fake.pdf",
                b"This is not a PDF file",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_upload_rejects_empty_pdf(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_ingests_valid_pdf_signature(client):
    store = api_main.app.state.document_store
    store.ingest_pdf.return_value = {
        "status": "ingested",
        "document_id": "doc-002",
        "filename": "uploaded.pdf",
        "page_count": 1,
        "chunk_count": 2,
    }

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "uploaded.pdf",
                b"%PDF-1.4\nmock PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ingested"
    assert response.json()["original_filename"] == "uploaded.pdf"
    store.ingest_pdf.assert_called_once()


def test_ask_uploaded_documents_returns_answer(client, monkeypatch):
    expected = {
        "question": "What does the report describe?",
        "answer": "It describes image authenticity verification.",
        "model": "mock-model",
        "retrieval_method": "phase6_uploaded_pdf_faiss",
        "generation_status": "success",
        "sources": [
            {
                "filename": "report.pdf",
                "page_number": 2,
            }
        ],
    }

    monkeypatch.setattr(
        api_main,
        "answer_from_uploaded_pdfs",
        Mock(return_value=expected),
    )

    response = client.post(
        "/api/v1/documents/ask",
        json={"question": "What does the report describe?"},
    )

    assert response.status_code == 200
    assert "image authenticity verification" in response.json()["answer"]
    assert response.json()["sources"][0]["filename"] == "report.pdf"


def test_document_endpoints_return_503_when_store_not_ready(client):
    api_main.app.state.documents_ready = False

    response = client.get("/api/v1/documents")

    assert response.status_code == 503
