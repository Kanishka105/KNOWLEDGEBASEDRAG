
import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[4]
INPUT_PATH = ROOT_DIR / "outputs" / "phase4" / "rag_answer.json"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase5"
OUTPUT_PATH = OUTPUT_DIR / "citation_evaluation.json"


CITATION_PATTERN = re.compile(
    r"\[Source\s+(\d+)\]",
    flags=re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def split_into_claim_candidates(answer: str) -> list[str]:
    """
    Split answer text into rough sentence-level claim candidates.

    This is a simple heuristic. It is not a linguistic parser,
    and abbreviations or decimal numbers may affect splitting.
    """

    answer_without_bullets = re.sub(
        r"(?m)^\s*(?:[-*]|\d+[.)])\s+",
        "",
        answer,
    )

    pieces = re.split(r"(?<=[.!?])\s+|\n+", answer_without_bullets)

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def get_sources_by_label(sources: list[dict]) -> dict[str, dict]:
    """Build a mapping from Source labels to source metadata."""

    mapping = {}

    for source in sources:
        label = source.get("source_label")
        document_id = source.get("document_id")
        evidence = source.get("evidence", "")

        if not isinstance(label, str) or not label.strip():
            continue

        mapping[label] = {
            **source,
            "document_id": (
                str(document_id) if document_id is not None else None
            ),
            "evidence": evidence if isinstance(evidence, str) else "",
        }

    return mapping


def verify_citations(response: dict) -> dict:
    """
    Verify citation identity and produce claim-level review candidates.

    Automated checks:
      - Citation labels resolve to retrieved sources.
      - Source metadata contains a document ID and evidence text.
      - Claims have citations or are marked as needing review.

    This does NOT automatically prove semantic entailment.
    """

    answer = response.get("answer", "")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The response has no non-empty answer.")

    sources = response.get("sources", [])

    if not isinstance(sources, list):
        raise ValueError("'sources' must be a list.")

    source_map = get_sources_by_label(sources)

    claims = split_into_claim_candidates(answer)
    claim_results = []

    for claim in claims:
        cited_numbers = CITATION_PATTERN.findall(claim)

        cited_labels = [
            f"Source {number}"
            for number in cited_numbers
        ]

        # Keep unique labels while preserving their order.
        cited_labels = list(dict.fromkeys(cited_labels))

        valid_labels = [
            label for label in cited_labels
            if label in source_map
        ]

        invalid_labels = [
            label for label in cited_labels
            if label not in source_map
        ]

        evidence_checks = []

        for label in valid_labels:
            source = source_map[label]

            evidence_checks.append({
                "source_label": label,
                "document_id": source["document_id"],
                "title": source.get("title", ""),
                "has_document_id": bool(source["document_id"]),
                "has_evidence_text": bool(
                    source["evidence"].strip()
                ),
            })

        if not cited_labels:
            status = "needs_citation_review"
        elif invalid_labels:
            status = "invalid_citation_label"
        elif any(
            not item["has_document_id"]
            or not item["has_evidence_text"]
            for item in evidence_checks
        ):
            status = "source_metadata_incomplete"
        else:
            # Identity is verified, but semantic support is not.
            status = "citation_resolves_support_unverified"

        claim_results.append({
            "claim_candidate": claim,
            "cited_source_labels": cited_labels,
            "valid_source_labels": valid_labels,
            "invalid_source_labels": invalid_labels,
            "source_evidence_checks": evidence_checks,
            "status": status,
        })

    total_claims = len(claim_results)

    cited_claims = sum(
        bool(item["cited_source_labels"])
        for item in claim_results
    )

    unresolved_claims = sum(
        item["status"] == "invalid_citation_label"
        for item in claim_results
    )

    claims_needing_review = sum(
        item["status"] in {
            "needs_citation_review",
            "source_metadata_incomplete",
            "citation_resolves_support_unverified",
        }
        for item in claim_results
    )

    return {
        "question": response.get("question", ""),
        "model": response.get("model", ""),
        "source_count": len(sources),
        "claim_candidate_count": total_claims,
        "claims_with_citations": cited_claims,
        "claims_without_citations": total_claims - cited_claims,
        "invalid_citation_claim_count": unresolved_claims,
        "claims_requiring_review": claims_needing_review,
        "semantic_support_verified": False,
        "important_note": (
            "This report checks citation identity and source metadata. "
            "It does not determine whether a cited passage logically "
            "or scientifically supports the claim. Review flagged "
            "claims against their evidence or use a separately "
            "validated claim-verification evaluator."
        ),
        "claim_results": claim_results,
        "sources": [
            {
                "source_label": source.get("source_label"),
                "document_id": source.get("document_id"),
                "title": source.get("title", ""),
                "evidence": source.get("evidence", ""),
            }
            for source in sources
        ],
    }


def main():
    response = load_json(INPUT_PATH)

    report = verify_citations(response)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== CITATION VERIFICATION ==========")
    print(f"Question: {report['question']}")
    print(f"Source count: {report['source_count']}")
    print(f"Claim candidates: {report['claim_candidate_count']}")
    print(f"Claims with citations: {report['claims_with_citations']}")
    print(f"Claims without citations: {report['claims_without_citations']}")
    print(
        "Claims with invalid citation labels:",
        report["invalid_citation_claim_count"],
    )
    print(
        "Semantic support verified:",
        report["semantic_support_verified"],
    )
    print(f"Saved report: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
