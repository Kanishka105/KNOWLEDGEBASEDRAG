
import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_PATH = PROJECT_ROOT / "outputs" / "phase5" / "retrieval_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase6"
OUTPUT_PATH = OUTPUT_DIR / "retrieval_error_analysis.json"

METHODS = ("bm25", "dense", "hybrid")


def analyze_query(query_record: dict) -> dict:
    """
    Inspect each method's top-k retrieval results using the per-query
    information saved by the Phase 5 evaluator.
    """
    query_id = query_record.get("query_id")
    query = query_record.get("query", "")
    relevant_count = query_record.get("relevant_document_count", 0)
    methods = query_record.get("methods", {})

    method_results = {}

    for method in METHODS:
        result = methods.get(method, {})
        retrieved_ids = result.get("retrieved_document_ids", [])

        # These IDs are needed to identify which relevant documents
        # were missed. Phase 5 must expose relevant IDs for this check.
        relevant_ids = query_record.get("relevant_document_ids")

        if relevant_ids is None:
            # Do not pretend that the missing IDs are known.
            method_results[method] = {
                "retrieved_count": len(retrieved_ids),
                "relevant_document_count": relevant_count,
                "retrieved_document_ids": retrieved_ids,
                "relevant_ids_available": False,
                "error_classification": (
                    "Cannot determine exact misses: "
                    "relevant_document_ids are not saved for this query."
                ),
            }
            continue

        relevant_set = {str(doc_id) for doc_id in relevant_ids}
        retrieved_set = {str(doc_id) for doc_id in retrieved_ids}

        hits = [
            doc_id for doc_id in retrieved_ids
            if str(doc_id) in relevant_set
        ]
        missed = sorted(relevant_set - retrieved_set)
        false_positives = [
            doc_id for doc_id in retrieved_ids
            if str(doc_id) not in relevant_set
        ]

        method_results[method] = {
            "retrieved_count": len(retrieved_ids),
            "relevant_document_count": len(relevant_set),
            "relevant_hits": hits,
            "missed_relevant_document_ids": missed,
            "nonrelevant_retrieved_document_ids": false_positives,
            "hit_count": len(hits),
            "relevant_ids_available": True,
        }

    hybrid = method_results.get("hybrid", {})
    hybrid_has_ids = hybrid.get("relevant_ids_available", False)

    if hybrid_has_ids:
        if hybrid.get("hit_count", 0) == 0:
            classification = "hybrid_missed_all_relevant_documents"
        elif hybrid.get("missed_relevant_document_ids"):
            classification = "hybrid_missed_some_relevant_documents"
        elif hybrid.get("nonrelevant_retrieved_document_ids"):
            classification = "hybrid_found_relevant_but_also_irrelevant"
        else:
            classification = "all_retrieved_documents_relevant"
    else:
        classification = "requires_relevance_id_data"

    return {
        "query_id": query_id,
        "query": query,
        "relevant_document_count": relevant_count,
        "hybrid_error_classification": classification,
        "methods": method_results,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Phase 5 metrics file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    query_records = data.get("per_query", [])
    if not query_records:
        raise ValueError(
            "No per_query records found in retrieval_metrics.json."
        )

    analyzed = [analyze_query(record) for record in query_records]

    classification_counts = Counter(
        record["hybrid_error_classification"] for record in analyzed
    )

    report = {
        "phase": "6.1",
        "title": "Retrieval Error Analysis",
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "queries_analyzed": len(analyzed),
        "baseline_summary": data.get("summary", {}),
        "classification_counts": dict(classification_counts),
        "relevance_id_limitation": (
            "Exact missed and nonrelevant document analysis requires "
            "relevant_document_ids in each per_query record. If those IDs "
            "are absent, the script explicitly marks the analysis as "
            "incomplete rather than guessing."
        ),
        "per_query": analyzed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("PHASE 6.1 — RETRIEVAL ERROR ANALYSIS")
    print("=" * 60)
    print(f"Queries analyzed: {len(analyzed)}")
    print("\nHybrid classifications:")
    for label, count in classification_counts.items():
        print(f"  {label}: {count}")
    print(f"\nSaved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
