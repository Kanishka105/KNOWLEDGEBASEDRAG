
from app.ai.rag.phase10_agentic_research.agent import evaluate_evidence


class FakeClient:
    class interactions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("503 Service Unavailable: high demand")


def test_evidence_fallback():
    result = evaluate_evidence(
        question="What evidence supports CBT for depression?",
        subquestions=[
            "What evidence supports CBT for depression?"
        ],
        sources=[
            {
                "source_label": "[S1]",
                "title": "Example research paper",
                "abstract": "Example abstract for testing.",
            }
        ],
        client=FakeClient(),
    )

    assert result["sufficient"] is False
    assert result["gaps"] == []
    assert (
        result["reason"]
        == "Evidence evaluation could not be completed."
    )

    print("Evidence fallback successful:", result)


if __name__ == "__main__":
    test_evidence_fallback()
