
import argparse
import json
import re
from pathlib import Path

from app.ai.rag.phase4_rag_generation.rag_pipeline import (
    create_gemini_client,
    load_retrieval_components,
    answer_question,
)
from app.ai.rag.phase1_dataset.dataset_loader import load_scifact


ROOT_DIR = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase5"

CHECKPOINT_PATH = OUTPUT_DIR / "generation_evaluation_checkpoint.json"
REPORT_PATH = OUTPUT_DIR / "generation_evaluation.json"


def content_tokens(text: str) -> set[str]:
    """Extract lowercase words for a simple lexical-overlap diagnostic."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def evaluate_answer(response: dict) -> dict:
    """
    Calculate basic generation diagnostics.

    Token overlap is only a lexical heuristic. It does not prove
    that the answer's claims are supported by the evidence.
    """

    answer = response.get("answer", "")
    sources = response.get("sources", [])
    citation_report = response.get("citation_validation", {})

    answer_tokens = content_tokens(answer)

    evidence_text = " ".join(
        source.get("evidence", "")
        for source in sources
    )
    evidence_tokens = content_tokens(evidence_text)

    overlap_ratio = (
        len(answer_tokens & evidence_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )

    return {
        "answer_nonempty": bool(answer.strip()),
        "answer_character_count": len(answer),
        "source_count": len(sources),
        "has_citations": citation_report.get("has_citations", False),
        "all_citations_resolve": citation_report.get(
            "all_citations_resolve", False
        ),
        "cited_source_labels": citation_report.get(
            "cited_source_labels", []
        ),
        "evidence_token_overlap_ratio": round(overlap_ratio, 4),
        "evaluation_note": (
            "Token overlap is a diagnostic heuristic, not a "
            "faithfulness or factual-correctness score."
        ),
    }


def get_question(query_id, queries) -> str:
    """Read a query from the dataset, supporting string and dict formats."""

    raw_query = queries.get(str(query_id))

    if raw_query is None:
        raw_query = queries.get(query_id)

    if isinstance(raw_query, dict):
        raw_query = raw_query.get(
            "text",
            raw_query.get("query", raw_query.get("query_text", "")),
        )

    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError(f"Invalid query text for query ID {query_id}.")

    return raw_query.strip()


def save_json(path: Path, payload: dict) -> None:
    """Write JSON with readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_report(records: list[dict], requested_queries: int) -> dict:
    """Build a report from the successfully completed queries."""

    count = len(records)

    report = {
        "dataset": "SciFact",
        "split": "test",
        "requested_queries": requested_queries,
        "evaluated_queries": count,
        "evaluation_type": "Basic generation diagnostics",
        "records": records,
    }

    if count:
        report["summary"] = {
            "nonempty_answer_rate": round(
                sum(
                    r["diagnostics"]["answer_nonempty"]
                    for r in records
                ) / count,
                4,
            ),
            "citation_presence_rate": round(
                sum(
                    r["diagnostics"]["has_citations"]
                    for r in records
                ) / count,
                4,
            ),
            "resolvable_citation_rate": round(
                sum(
                    r["diagnostics"]["all_citations_resolve"]
                    for r in records
                ) / count,
                4,
            ),
            "mean_evidence_token_overlap": round(
                sum(
                    r["diagnostics"]["evidence_token_overlap_ratio"]
                    for r in records
                ) / count,
                4,
            ),
        }

    return report


def run_evaluation(
    max_queries: int = 5,
    resume: bool = False,
) -> dict:
    if max_queries < 1:
        raise ValueError("--max-queries must be positive.")

    corpus, queries, qrels = load_scifact(split="test")

    # Include only queries that have relevance judgments and query text.
    query_ids = [
        str(query_id)
        for query_id in qrels
        if str(query_id) in queries or query_id in queries
    ][:max_queries]

    if not query_ids:
        raise ValueError(
            "No queries matched the available relevance judgments."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    completed_ids = set()

    # Optionally resume completed queries from a previous checkpoint.
    if resume and CHECKPOINT_PATH.exists():
        previous = json.loads(
            CHECKPOINT_PATH.read_text(encoding="utf-8")
        )

        records = previous.get("records", [])
        completed_ids = {
            str(record["query_id"])
            for record in records
        }

        print(f"Resuming with {len(records)} saved records.")

    remaining_ids = [
        query_id
        for query_id in query_ids
        if query_id not in completed_ids
    ]

    print(f"Queries requested: {len(query_ids)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(remaining_ids)}")

    if not remaining_ids:
        report = build_report(records, max_queries)
        save_json(REPORT_PATH, report)
        print(f"All requested queries are already complete: {REPORT_PATH}")
        return report

    client = create_gemini_client()
    components = load_retrieval_components()

    try:
        for position, query_id in enumerate(remaining_ids, start=1):
            question = get_question(query_id, queries)

            print(
                f"\n[{position}/{len(remaining_ids)}] "
                f"Query ID: {query_id}"
            )
            print(f"Question: {question}")

            try:
                response = answer_question(
                    question=question,
                    components=components,
                    client=client,
                )

                diagnostics = evaluate_answer(response)

                record = {
                    "query_id": query_id,
                    "question": question,
                    "answer": response["answer"],
                    "sources": response["sources"],
                    "diagnostics": diagnostics,
                }

                records.append(record)
                completed_ids.add(query_id)

                # Save after every successful query.
                checkpoint = build_report(records, max_queries)
                checkpoint["status"] = "in_progress"
                save_json(CHECKPOINT_PATH, checkpoint)

                print("Answer generated and checkpoint saved.")
                print(
                    "Citation labels resolve:",
                    diagnostics["all_citations_resolve"],
                )
                print(
                    "Evidence token overlap:",
                    diagnostics["evidence_token_overlap_ratio"],
                )

            except KeyboardInterrupt:
                print("\nEvaluation interrupted by user.")
                print("Completed query results are saved in the checkpoint.")
                raise

            except Exception as exc:
                # Record the failure and continue with the next query.
                print(
                    f"Query {query_id} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                failure_record = {
                    "query_id": query_id,
                    "question": question,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

                checkpoint = build_report(records, max_queries)
                checkpoint["status"] = "in_progress"
                checkpoint.setdefault("failed_queries", []).append(
                    failure_record
                )
                save_json(CHECKPOINT_PATH, checkpoint)

    except KeyboardInterrupt:
        # Preserve progress even if interrupted during an API request.
        checkpoint = build_report(records, max_queries)
        checkpoint["status"] = "interrupted"
        save_json(CHECKPOINT_PATH, checkpoint)
        print(f"Checkpoint saved to: {CHECKPOINT_PATH.resolve()}")
        return checkpoint

    report = build_report(records, max_queries)
    report["status"] = (
        "completed" if len(records) == len(query_ids) else "partial"
    )

    save_json(REPORT_PATH, report)

    # Retain a final checkpoint too.
    save_json(CHECKPOINT_PATH, report)

    print("\n========== GENERATION EVALUATION ==========")
    print(f"Successfully evaluated: {len(records)}/{len(query_ids)}")
    print(f"Final report: {REPORT_PATH.resolve()}")

    if "summary" in report:
        print(json.dumps(report["summary"], indent=2))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 4 RAG answer generation."
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help="Number of queries to evaluate.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved generation checkpoint.",
    )

    args = parser.parse_args()

    run_evaluation(
        max_queries=args.max_queries,
        resume=args.resume,
    )
