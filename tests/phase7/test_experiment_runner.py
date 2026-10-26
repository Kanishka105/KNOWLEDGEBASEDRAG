
import json

import pytest

from app.ai.rag.phase7_advanced_retrieval.evaluation import (
    experiment_runner as runner,
)


def test_save_and_compare_experiments(tmp_path, monkeypatch):
    report_path = tmp_path / "experiment_results.json"

    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(runner, "REPORT_PATH", report_path)

    runner.save_experiment(
        name="BM25 baseline",
        method="bm25",
        parameters={"k": 5},
        metrics={"ndcg@5": 0.71, "mrr@5": 0.69},
    )

    runner.save_experiment(
        name="Dense baseline",
        method="dense",
        parameters={"k": 5},
        metrics={"ndcg@5": 0.67, "mrr@5": 0.63},
    )

    ranked = runner.compare_experiments("ndcg@5")

    assert ranked[0]["name"] == "BM25 baseline"
    assert ranked[1]["name"] == "Dense baseline"

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(saved["experiments"]) == 2


def test_save_experiment_rejects_empty_name():
    with pytest.raises(ValueError):
        runner.save_experiment(
            name=" ",
            method="bm25",
            parameters={},
            metrics={"ndcg@5": 0.5},
        )


def test_compare_returns_empty_when_metric_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(
        runner, "REPORT_PATH", tmp_path / "experiment_results.json"
    )

    runner.save_experiment(
        name="BM25 baseline",
        method="bm25",
        parameters={},
        metrics={"mrr@5": 0.5},
    )

    assert runner.compare_experiments("ndcg@5") == []
