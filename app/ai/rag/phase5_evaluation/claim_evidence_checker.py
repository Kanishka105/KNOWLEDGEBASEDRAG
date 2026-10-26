
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
INPUT_PATH = ROOT_DIR / "outputs" / "phase5" / "citation_evaluation.json"
OUTPUT_PATH = ROOT_DIR / "outputs" / "phase5" / "claim_evidence_review.json"


def tokens(text: str) -> set[str]:
    """Extract lowercase words, excluding very short tokens."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def check_claims(report: dict) -> dict:
    reviewed_claims = []

    for item in report.get("claim_results", []):
        claim = item.get("claim_candidate", "")
        claim_tokens = tokens(claim)

        source_reviews = []

        for source in item.get("source_evidence_checks", []):
            label = source["source_label"]

            matching_source = next(
                (
                    s for s in report.get("sources", [])
                    if s.get("source_label") == label
                ),
                None,
            )

            evidence = (
                matching_source.get("evidence", "")
                if matching_source else ""
            )

            evidence_tokens = tokens(evidence)

            overlap = (
                len(claim_tokens & evidence_tokens) / len(claim_tokens)
                if claim_tokens else 0.0
            )

            source_reviews.append({
                "source_label": label,
                "document_id": source.get("document_id"),
                "title": source.get("title", ""),
                "lexical_overlap": round(overlap, 4),
                "evidence_excerpt": evidence[:500],
                "semantic_support": "NOT_VERIFIED",
            })

        reviewed_claims.append({
            "claim": claim,
            "citation_status": item.get("status"),
            "source_reviews": source_reviews,
            "review_required": (
                not source_reviews
                or any(
                    source["lexical_overlap"] < 0.2
                    for source in source_reviews
                )
            ),
        })

    return {
        "method": "Lexical overlap diagnostic",
        "semantic_support_verified": False,
        "note": (
            "Lexical overlap can flag possible evidence mismatches, "
            "but cannot establish entailment or scientific correctness. "
            "Review claims and passages before assigning support labels."
        ),
        "claim_count": len(reviewed_claims),
        "claims_requiring_manual_review": sum(
            item["review_required"] for item in reviewed_claims
        ),
        "claims": reviewed_claims,
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Run citation_checker first: {INPUT_PATH}"
        )

    report = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    result = check_claims(report)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== CLAIM-EVIDENCE REVIEW ==========")
    print("Claims checked:", result["claim_count"])
    print(
        "Claims flagged for manual review:",
        result["claims_requiring_manual_review"],
    )
    print("Semantic support verified: False")
    print("Saved report:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
