
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT_DIR = Path(__file__).resolve().parents[4]
PHASE5_DIR = ROOT_DIR / "outputs" / "phase5"
OUTPUT_PATH = PHASE5_DIR / "final_evaluation_report.json"


REPORT_FILES = {
    "retrieval": PHASE5_DIR / "retrieval_metrics.json",
    "generation": PHASE5_DIR / "generation_evaluation.json",
    "generation_checkpoint": PHASE5_DIR / "generation_evaluation_checkpoint.json",
    "citation": PHASE5_DIR / "citation_evaluation.json",
    "lexical_claim_review": PHASE5_DIR / "claim_evidence_review.json",
    "semantic_faithfulness": PHASE5_DIR / "semantic_faithfulness.json",
}


def read_json(path: Path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"read_error": str(exc)}


def main():
    PHASE5_DIR.mkdir(parents=True, exist_ok=True)

    reports = {
        name: read_json(path)
        for name, path in REPORT_FILES.items()
    }

    # Prefer the final generation report when available.
    generation = (
        reports["generation"]
        if reports["generation"] is not None
        else reports["generation_checkpoint"]
    )

    if reports["generation"] is None and generation is not None:
        generation_status = generation.get("status", "unknown")
    elif generation is not None:
        generation_status = generation.get("status", "unknown")
    else:
        generation_status = "missing"

    retrieval = reports["retrieval"]
    citation = reports["citation"]
    lexical = reports["lexical_claim_review"]
    semantic = reports["semantic_faithfulness"]

    summary = {
        "retrieval_report_available": retrieval is not None,
        "generation_report_available": reports["generation"] is not None,
        "generation_status": generation_status,
        "generation_queries_evaluated": (
            generation.get("evaluated_queries")
            if isinstance(generation, dict)
            else None
        ),
        "citation_report_available": citation is not None,
        "lexical_claim_review_available": lexical is not None,
        "semantic_faithfulness_report_available": semantic is not None,
        "semantic_support_proven": False,
    }

    final_report = {
        "project": "SciFact Retrieval-Augmented Generation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "reports": reports,
        "limitations": [
            "Citation-label validity does not prove claim support.",
            "Lexical overlap is a heuristic, not a faithfulness score.",
            "NLI predictions require validation against manually reviewed claims.",
            "Missing reports must not be interpreted as successful evaluations.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== FINAL PHASE 5 REPORT ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved report: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
