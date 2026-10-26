
import json
from pathlib import Path
from typing import Any

from app.ai.rag.phase7_advanced_retrieval.evaluation.experiment_runner import (
    save_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
BASELINE_PATH = PROJECT_ROOT / "outputs" / "phase5" / "retrieval_metrics.json"


def import_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline report not found: {BASELINE_PATH}. "
            "Wait for the Phase 5 evaluation to finish first."
        )

    with BASELINE_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    summary = report.get("summary", {})
    if not summary:
        raise ValueError("Baseline report contains no summary metrics.")

    for method, metrics in summary.items():
        normalized_metrics = {
            str(name).lower(): float(value)
            for name, value in metrics.items()
        }

        save_experiment(
            name=f"Phase 5 baseline — {method.upper()}",
            method=method,
            parameters={
                "source": "phase5_evaluation",
                "dataset": report.get("dataset", "SciFact"),
                "split": report.get("split", "test"),
                "k": report.get("k", 5),
                "evaluated_query_count": report.get(
                    "evaluated_query_count", 0
                ),
            },
            metrics=normalized_metrics,
            notes="Imported measured results; no retrieval was rerun.",
        )

    print(f"Imported {len(summary)} retrieval methods.")
    print(f"Source: {BASELINE_PATH}")
    return report


if __name__ == "__main__":
    import_baseline()
