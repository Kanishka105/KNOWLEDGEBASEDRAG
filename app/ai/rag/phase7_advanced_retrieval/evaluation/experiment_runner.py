
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE7_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = PHASE7_DIR / "reports"
REPORT_PATH = REPORT_DIR / "experiment_results.json"


def load_report() -> dict[str, Any]:
    """Load the experiment report or create an empty one."""
    if not REPORT_PATH.exists():
        return {
            "project": "SciFact Advanced Retrieval Experiments",
            "dataset": "SciFact",
            "experiments": [],
        }

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("Experiment report must contain a JSON object.")

    report.setdefault("experiments", [])
    return report


def save_experiment(
    name: str,
    method: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    notes: str = "",
) -> dict[str, Any]:
    """Record one completed experiment with its configuration and metrics."""
    if not name.strip():
        raise ValueError("Experiment name cannot be empty.")

    if not method.strip():
        raise ValueError("Retrieval method cannot be empty.")

    if not metrics:
        raise ValueError("Metrics cannot be empty.")

    record = {
        "name": name,
        "method": method,
        "parameters": parameters,
        "metrics": metrics,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = load_report()
    report["experiments"].append(record)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    return record


def compare_experiments(
    metric: str = "ndcg@5",
) -> list[dict[str, Any]]:
    """Return experiments ranked by a metric, highest first."""
    report = load_report()
    experiments = [
        experiment
        for experiment in report["experiments"]
        if metric in experiment.get("metrics", {})
    ]

    return sorted(
        experiments,
        key=lambda experiment: experiment["metrics"][metric],
        reverse=True,
    )


if __name__ == "__main__":
    ranked = compare_experiments()

    if not ranked:
        print("No recorded experiments contain the selected metric.")
    else:
        print("Experiments ranked by nDCG@5:")

        for experiment in ranked:
            print(
                f'{experiment["name"]}: '
                f'{experiment["metrics"]["ndcg@5"]:.4f}'
            )
