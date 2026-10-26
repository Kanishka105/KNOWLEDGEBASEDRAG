
from app.ai.rag.phase10_agentic_research.agent import plan_research


class FakeClient:
    class interactions:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError(
                "503 Service Unavailable: high demand"
            )


def test_planner_fallback():
    question = (
        "What evidence supports cognitive behavioral "
        "therapy for depression?"
    )

    result = plan_research(question, FakeClient())

    assert result == [question]
    print("Fallback successful:", result == [question])
    print("Plan:", result)


if __name__ == "__main__":
    test_planner_fallback()
