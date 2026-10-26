# Phase 7 — Advanced Retrieval Experiments

## Objective
Evaluate Reciprocal Rank Fusion (RRF) parameter choices for hybrid retrieval
using the SciFact test dataset.

## Experimental setup
- Dataset: SciFact
- Split: test
- Queries: 300
- Evaluation depth: top 5
- RRF parameters: k = 20, 40, 60, 100
- Baselines: BM25, dense retrieval, hybrid retrieval

## Results

| Method | Precision@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.1580 | 0.7284 | 0.6132 | 0.6360 |
| Dense | 0.1640 | 0.7379 | 0.5997 | 0.6293 |
| Hybrid baseline | 0.1653 | 0.7482 | 0.6528 | 0.6678 |
| RRF k=20 | 0.1653 | 0.7482 | 0.6514 | 0.6673 |
| RRF k=40 | 0.1653 | 0.7482 | 0.6528 | 0.6678 |
| RRF k=60 | 0.1653 | 0.7482 | 0.6528 | 0.6678 |
| RRF k=100 | 0.1647 | 0.7474 | 0.6528 | 0.6670 |

## Findings
- Hybrid retrieval outperformed the individual BM25 and dense baselines
  on MRR@5 and nDCG@5.
- RRF k=40 and k=60 matched the hybrid baseline on the displayed metrics.
- The tested values of k produced similar results.
- These experiments do not demonstrate a meaningful improvement from tuning k.

## Limitations
- Verify that every method used the same query IDs and metric definitions.
- Matching aggregate metrics do not prove that the ranked document lists are identical.
- Results apply to this SciFact test evaluation and should not be generalized
  to other datasets without additional testing.

## Reproducibility
Machine-readable results:
reports/final_comparison.json

Experiment history:
reports/experiment_results.json

Phase 7 tests:
python -m pytest tests/phase7 -v
