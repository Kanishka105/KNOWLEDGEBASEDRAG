
"""Bounded agentic research workflow built on the existing Phase 4 RAG pipeline."""

import json
import logging
import re
from typing import Any

from app.ai.rag.phase4_rag_generation.rag_pipeline import (
    GEMINI_MODEL,
    build_context,
    generate_answer,
    retrieve_sources,
    validate_citations,
)

logger = logging.getLogger(__name__)

MAX_SUBQUESTIONS = 3
MAX_RETRIEVAL_ROUNDS = 2


def _call_gemini(client: Any, prompt: str) -> str:
    """Call the same Gemini Interactions API used by Phase 4."""
    response = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    output = response.output_text

    if not output or not output.strip():
        raise RuntimeError("Gemini returned an empty agent response.")

    return output.strip()


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object, allowing optional Markdown code fences."""
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError("Agent response did not contain a JSON object.")

    result = json.loads(cleaned[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("Agent response must be a JSON object.")

    return result


def plan_research(question: str, client: Any) -> list[str]:
    """Plan research, falling back to the original question if planning fails."""

    prompt = f"""
You are a scientific research planning assistant.

Break the user's question into at most {MAX_SUBQUESTIONS} focused,
non-duplicate subquestions that can be answered using scientific papers.

Return JSON only:
{{"subquestions": ["question one", "question two"]}}

Keep the original question's intent. Do not answer the questions.
If the original question is already narrow, return it as the only subquestion.

User question:
{question}
"""

    try:
        result = _parse_json_object(_call_gemini(client, prompt))
        items = result.get("subquestions", [])

        if not isinstance(items, list):
            raise ValueError("'subquestions' must be a list.")

        questions = [
            item.strip()
            for item in items
            if isinstance(item, str) and item.strip()
        ]

        # Remove duplicates and respect the configured limit.
        questions = list(dict.fromkeys(questions))[:MAX_SUBQUESTIONS]

        if not questions:
            raise ValueError("Planner returned no usable subquestions.")

        return questions

    except Exception as exc:
        logger.warning(
            "Planning failed; using the original question. Error type: %s",
            type(exc).__name__,
        )
        return [question]


def evaluate_evidence(
    question: str,
    subquestions: list[str],
    sources: list[dict],
    client: Any,
) -> dict:
    """Evaluate evidence coverage without crashing if the evaluator fails."""

    if not sources:
        return {
            "sufficient": False,
            "gaps": subquestions[:MAX_SUBQUESTIONS],
            "reason": "No sources were retrieved.",
        }

    source_summary = "\n".join(
        f"- {source.get('source_label', 'Unlabelled source')}: "
        f"{source.get('title', 'Untitled paper')} "
        f"{source.get('abstract', source.get('text', ''))}"
        for source in sources
    )

    prompt = f"""
Evaluate whether the retrieved scientific evidence covers the question.

Original question:
{question}

Subquestions:
{json.dumps(subquestions, ensure_ascii=False)}

Retrieved evidence:
{source_summary[:18000]}

Return JSON only:
{{
  "sufficient": true,
  "gaps": [],
  "reason": "Brief explanation"
}}

Mark sufficient true only if the evidence reasonably covers the question.
If important aspects remain unanswered, list those as gaps.
Do not invent sources or claim that a source supports something it does not.
"""

    try:
        result = _parse_json_object(_call_gemini(client, prompt))

        gaps = result.get("gaps", [])

        if not isinstance(gaps, list):
            gaps = []

        clean_gaps = [
            gap.strip()
            for gap in gaps
            if isinstance(gap, str) and gap.strip()
        ][:MAX_SUBQUESTIONS]

        return {
            "sufficient": result.get("sufficient") is True,
            "gaps": clean_gaps,
            "reason": str(result.get("reason", "")),
        }

    except Exception as exc:
        logger.warning(
            "Evidence evaluation failed. Error type: %s",
            type(exc).__name__,
        )

        # Fail safely: do not claim that evidence is sufficient.
        # Empty gaps prevent another retrieval round based on
        # an evaluation that could not be completed.
        return {
            "sufficient": False,
            "gaps": [],
            "reason": "Evidence evaluation could not be completed.",
        }


def _deduplicate_sources(results: list[dict]) -> list[dict]:
    """Remove repeated papers while preserving retrieval order."""
    unique = []
    seen = set()

    for result in results:
        identifier = (
            result.get("document_id")
            or result.get("doc_id")
            or result.get("id")
            or result.get("title")
        )

        if identifier is None:
            identifier = json.dumps(result, sort_keys=True, default=str)

        key = str(identifier).strip().casefold()

        if key and key not in seen:
            seen.add(key)
            unique.append(result)

    return unique


def run_agentic_research(
    question: str,
    components: dict,
    client: Any,
) -> dict:
    """Plan, retrieve, evaluate, and generate a cited research answer."""
    question = question.strip()

    if not question:
        raise ValueError("Please provide a research question.")

    logger.info("Starting agentic research workflow")

    # 1. Plan the research.
    subquestions = plan_research(question, client)
    pending_questions = list(subquestions)

    # 2. Retrieve evidence in a bounded loop.
    all_results = []
    retrieval_history = []

    evaluation = {
        "sufficient": False,
        "gaps": pending_questions,
        "reason": "Evidence has not yet been evaluated.",
    }

    for round_number in range(1, MAX_RETRIEVAL_ROUNDS + 1):
        if not pending_questions:
            break

        round_questions = pending_questions[:MAX_SUBQUESTIONS]
        pending_questions = []

        for subquestion in round_questions:
            try:
                results = retrieve_sources(subquestion, components)

                if results:
                    all_results.extend(results)

                retrieval_history.append(
                    {
                        "round": round_number,
                        "question": subquestion,
                        "retrieved_count": len(results),
                        "status": "success",
                    }
                )

            except (RuntimeError, ValueError, KeyError) as exc:
                logger.warning(
                    "Retrieval failed for a subquestion. "
                    "Error type: %s",
                    type(exc).__name__,
                )

                retrieval_history.append(
                    {
                        "round": round_number,
                        "question": subquestion,
                        "retrieved_count": 0,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )

        all_results = _deduplicate_sources(all_results)

        if not all_results:
            break

        # Prepare labelled evidence for evaluation.
        _, labelled_sources = build_context(all_results)

        evaluation = evaluate_evidence(
            question=question,
            subquestions=round_questions,
            sources=labelled_sources,
            client=client,
        )

        if evaluation["sufficient"]:
            break

        # Retrieve again only when the evaluator identifies
        # new, specific gaps.
        if round_number < MAX_RETRIEVAL_ROUNDS:
            current_questions = {
                item.casefold() for item in round_questions
            }

            pending_questions = [
                gap
                for gap in evaluation["gaps"]
                if gap.casefold() not in current_questions
            ][:MAX_SUBQUESTIONS]

    if not all_results:
        raise RuntimeError(
            "The agent could not retrieve evidence for this question."
        )

    # 3. Generate the final answer using the existing Phase 4 pipeline.
    # This step still requires Gemini to be available.
    context, sources = build_context(all_results)

    answer = generate_answer(
        question=question,
        context=context,
        client=client,
    )

    citation_report = validate_citations(answer, sources)

    logger.info(
        "Agentic research completed with %d unique sources",
        len(sources),
    )

    # 4. Return the answer and workflow trace.
    return {
        "question": question,
        "answer": answer,
        "model": GEMINI_MODEL,
        "workflow": "agentic_research",
        "plan": subquestions,
        "retrieval_history": retrieval_history,
        "evidence_evaluation": evaluation,
        "iterations_used": max(
            (item["round"] for item in retrieval_history),
            default=0,
        ),
        "sources": sources,
        "citation_validation": citation_report,
    }
