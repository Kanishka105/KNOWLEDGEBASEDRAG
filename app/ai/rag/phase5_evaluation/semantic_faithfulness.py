
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT_DIR = Path(__file__).resolve().parents[4]
INPUT_PATH = ROOT_DIR / "outputs" / "phase4" / "rag_answer.json"
OUTPUT_PATH = ROOT_DIR / "outputs" / "phase5" / "semantic_faithfulness.json"

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
MAX_LENGTH = 512


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence candidates and omit editorial notes."""

    answer = re.sub(
        r"(?is)\(?(?:\*{1,2})?\s*note\s*:.*?(?:\)\s*\*{0,2})?(?=\n|$)",
        "",
        answer,
    )

    pieces = re.split(r"(?<=[.!?])\s+|\n+", answer)

    return [
        piece.strip().lstrip("*- ").rstrip("*").strip()
        for piece in pieces
        if piece.strip()
    ]


def get_label_map(model) -> dict[str, int]:
    """Normalize model labels, including labels stored in model config."""

    mapping = {}

    for index, label in model.config.id2label.items():
        normalized = str(label).lower()
        mapping[normalized] = int(index)

    return mapping


def find_label(label_map: dict[str, int], keyword: str):
    for label, index in label_map.items():
        if keyword in label:
            return index
    return None


def classify_pair(claim: str, evidence: str, tokenizer, model, label_map):
    """Classify evidence as entailment, contradiction, or neutral."""

    encoded = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    model.eval()

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    entailment_id = find_label(label_map, "entail")
    contradiction_id = find_label(label_map, "contrad")
    neutral_id = find_label(label_map, "neutral")

    if any(
        label_id is None
        for label_id in (entailment_id, contradiction_id, neutral_id)
    ):
        raise ValueError(
            f"Cannot identify NLI labels from model config: {label_map}"
        )

    scores = {
        "entailment": float(probabilities[entailment_id]),
        "contradiction": float(probabilities[contradiction_id]),
        "neutral": float(probabilities[neutral_id]),
    }

    predicted = max(scores, key=scores.get)

    # These are model predictions, not calibrated truth probabilities.
    return {
        "predicted_relation": predicted,
        "scores": {key: round(value, 4) for key, value in scores.items()},
    }


def verify_response(response: dict, tokenizer, model, label_map) -> dict:
    answer = response.get("answer", "")
    sources = response.get("sources", [])

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The saved answer is empty.")

    source_map = {
        str(source.get("source_label", "")).strip(): source
        for source in sources
        if source.get("source_label")
    }

    claims = split_claims(answer)
    results = []

    for claim in claims:
        cited_numbers = re.findall(
            r"\[Source\s+(\d+)\]",
            claim,
            flags=re.IGNORECASE,
        )
        cited_labels = list(dict.fromkeys(
            f"Source {number}" for number in cited_numbers
        ))

        if not cited_labels:
            results.append({
                "claim": claim,
                "status": "needs_citation_review",
                "source_checks": [],
            })
            continue

        source_checks = []

        for label in cited_labels:
            source = source_map.get(label)

            if source is None:
                source_checks.append({
                    "source_label": label,
                    "status": "invalid_citation",
                })
                continue

            evidence = source.get("evidence", "")
            if not isinstance(evidence, str) or not evidence.strip():
                source_checks.append({
                    "source_label": label,
                    "document_id": source.get("document_id"),
                    "status": "missing_evidence",
                })
                continue

            prediction = classify_pair(
                claim=claim,
                evidence=evidence,
                tokenizer=tokenizer,
                model=model,
                label_map=label_map,
            )

            source_checks.append({
                "source_label": label,
                "document_id": source.get("document_id"),
                "title": source.get("title", ""),
                **prediction,
            })

        valid_predictions = [
            item for item in source_checks
            if item.get("predicted_relation")
        ]

        if not valid_predictions:
            status = "needs_review"
        elif any(
            item["predicted_relation"] == "entailment"
            for item in valid_predictions
        ):
            status = "model_predicts_supported"
        elif all(
            item.get("predicted_relation") == "contradiction"
            for item in valid_predictions
        ):
            status = "model_predicts_contradicted"
        else:
            status = "insufficient_or_mixed_evidence"

        results.append({
            "claim": claim,
            "cited_source_labels": cited_labels,
            "status": status,
            "source_checks": source_checks,
        })

    return {
        "question": response.get("question", ""),
        "model": MODEL_NAME,
        "claim_count": len(results),
        "semantic_support_verified": False,
        "note": (
            "Statuses are NLI model predictions, not verified facts. "
            "Review scientific claims and evaluate model accuracy "
            "against manually annotated examples."
        ),
        "claims": results,
    }


def main():
    response = load_json(INPUT_PATH)

    print(f"Loading NLI model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    label_map = get_label_map(model)

    print("Model labels:", label_map, flush=True)
    print("Evaluating cited claims...", flush=True)

    report = verify_response(response, tokenizer, model, label_map)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== SEMANTIC FAITHFULNESS ==========")
    print("Claims analyzed:", report["claim_count"])

    for item in report["claims"]:
        print(f"\nClaim: {item['claim'][:180]}")
        print(f"Status: {item['status']}")

        for source in item["source_checks"]:
            print(
                f"  {source['source_label']}: "
                f"{source.get('predicted_relation', source.get('status'))}"
            )

    print(f"\nSaved report: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
