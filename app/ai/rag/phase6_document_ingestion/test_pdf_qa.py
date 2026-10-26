
from unittest.mock import Mock

from app.ai.rag.phase6_document_ingestion.pdf_qa import (
    answer_from_uploaded_pdfs,
)


class FakeDocumentStore:
    def search(self, query, top_k=5):
        return [
            {
                "text": (
                    "The platform supports image authenticity "
                    "verification and before-and-after evidence checks."
                ),
                "document_id": "doc-1",
                "filename": "municipal-report.pdf",
                "page_number": 5,
                "similarity_score": 0.91,
            }
        ]


def test_quota_error_returns_retrieved_evidence_without_retry():
    client = Mock()

    client.models.generate_content.side_effect = RuntimeError(
        "RESOURCE_EXHAUSTED: free tier quota exceeded"
    )

    result = answer_from_uploaded_pdfs(
        question="What does the platform support?",
        document_store=FakeDocumentStore(),
        client=client,
        top_k=3,
    )

    assert (
        result["generation_status"]
        == "free_quota_exhausted_fallback"
    )
    assert "image authenticity verification" in result["answer"]
    assert result["sources"][0]["filename"] == "municipal-report.pdf"
    assert result["sources"][0]["page_number"] == 5

    # Ensure the Gemini generation call was attempted only once.
    assert client.models.generate_content.call_count == 1
