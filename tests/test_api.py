
from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["application"] == "SciFact Research Assistant"
    assert data["version"] == "1.4.0"
    assert data["docs"] == "/docs"
    assert data["endpoints"]["scientific_rag"] == "/api/v1/ask"
    assert data["endpoints"]["upload_pdf"] == "/api/v1/documents/upload"
    assert data["endpoints"]["list_documents"] == "/api/v1/documents"
    assert data["endpoints"]["search_documents"] == "/api/v1/documents/search"
    assert data["endpoints"]["ask_uploaded_documents"] == "/api/v1/documents/ask"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert "rag_ready" in response.json()


def test_question_too_short_is_rejected():
    response = client.post(
        "/api/v1/ask",
        json={"question": "AI"},
    )

    assert response.status_code == 422


def test_empty_question_is_rejected():
    response = client.post(
        "/api/v1/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_returns_503_when_rag_is_not_ready():
    app.state.rag_ready = False

    response = client.post(
        "/api/v1/ask",
        json={
            "question": "What evidence suggests biomaterials influence stem cells?"
        },
    )

    assert response.status_code == 503


def test_ask_returns_answer_when_rag_is_ready(monkeypatch):
    from app.api import main as api_main

    app.state.rag_ready = True
    app.state.rag_components = {"test": True}
    app.state.gemini_client = object()

    def fake_answer_question(question, components, client):
        return {
            "question": question,
            "answer": "Biomaterial geometry can influence cell behavior [Source 1].",
            "model": "test-model",
            "retrieval_method": "BM25 + FAISS + RRF + CrossEncoder",
            "sources": [
                {
                    "source_label": "Source 1",
                    "document_id": "15337254",
                    "title": "Geometric cues for directing differentiation",
                    "retrieval_rank": 1,
                    "evidence": "Geometry can influence cell behavior.",
                }
            ],
            "citation_validation": {
                "has_citations": True,
                "all_citations_resolve": True,
                "cited_source_labels": ["Source 1"],
                "valid_source_labels": ["Source 1"],
                "invalid_source_labels": [],
            },
        }

    monkeypatch.setattr(
        api_main,
        "run_rag_question",
        fake_answer_question,
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "question": "What evidence suggests biomaterials influence stem cells?"
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["sources"][0]["document_id"] == "15337254"
    assert data["citation_validation"]["all_citations_resolve"] is True
