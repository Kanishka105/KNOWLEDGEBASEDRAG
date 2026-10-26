
from unittest.mock import patch

from app.ai.rag.phase10_agentic_research import agent


def test_complete_agent_workflow():
    question = "What evidence supports CBT for depression?"

    fake_results = [
        {
            "document_id": "paper-1",
            "title": "CBT and Depression",
            "abstract": "A study examining cognitive behavioral therapy.",
        }
    ]

    fake_sources = [
        {
            "source_label": "[S1]",
            "title": "CBT and Depression",
            "abstract": "A study examining cognitive behavioral therapy.",
        }
    ]

    with (
        patch.object(
            agent,
            "plan_research",
            return_value=[question],
        ),
        patch.object(
            agent,
            "retrieve_sources",
            return_value=fake_results,
        ),
        patch.object(
            agent,
            "build_context",
            return_value=("Evidence context [S1].", fake_sources),
        ),
        patch.object(
            agent,
            "evaluate_evidence",
            return_value={
                "sufficient": True,
                "gaps": [],
                "reason": "Mock evidence is sufficient.",
            },
        ),
        patch.object(
            agent,
            "generate_answer",
            return_value="CBT has been studied for depression [S1].",
        ),
        patch.object(
            agent,
            "validate_citations",
            return_value={"valid": True},
        ),
    ):
        result = agent.run_agentic_research(
            question=question,
            components={},
            client=None,
        )

    assert result["question"] == question
    assert result["workflow"] == "agentic_research"
    assert result["plan"] == [question]
    assert result["answer"].endswith("[S1].")
    assert result["citation_validation"]["valid"] is True
    assert result["iterations_used"] == 1

    print("Complete workflow test: PASS")
    print("Workflow:", result["workflow"])
    print("Sources:", len(result["sources"]))
    print("Iterations:", result["iterations_used"])


if __name__ == "__main__":
    test_complete_agent_workflow()
