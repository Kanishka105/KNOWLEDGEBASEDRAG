
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = PROJECT_ROOT / "datasets" / "scifact"

# find location of a dataset file.
def find_dataset_file(filename: str) -> Path:
    """Find a SciFact file inside the extracted dataset directory."""
    matches = list(DATASET_DIR.rglob(filename))
    # DATASET_DIR.rglob(filename) searches inside the SciFact directory and its subdirectories.

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {DATASET_DIR}"
        )

    return matches[0]


def load_jsonl(path: Path) -> dict:
    """Load JSONL records into a dictionary indexed by record ID."""
    records = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
                #  line is string not dict. So it converts a JSON formatted string into a python object.
            record = json.loads(line)
            record_id = str(
                record.get("_id", record.get("id", ""))
            )

            if not record_id:
                raise ValueError(
                    f"Missing record ID in {path.name}, "
                    f"line {line_number}"
                )

            if record_id in records:
                raise ValueError(
                    f"Duplicate ID {record_id} in {path.name}"
                )

            records[record_id] = record

    return records


def load_qrels(path: Path) -> dict:
    """Load relevance judgments, handling TSV headers safely."""

    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    # Detect whether the first row is a header.
    first_columns = first_line.lower().split("\t")
    has_header = (
        "score" in first_columns
        or "query_id" in first_columns
        or "corpus_id" in first_columns
    )

    if has_header:
        frame = pd.read_csv(path, sep="\t", dtype=str)
        frame.columns = [
            str(column).strip().lower()
            for column in frame.columns
        ]

        # Support common alternative column names.
        frame = frame.rename(
            columns={
                "query-id": "query_id",
                "corpus-id": "corpus_id",
                "doc_id": "corpus_id",
                "document_id": "corpus_id",
                "relevance": "score",
            }
        )

        required = ["query_id", "corpus_id", "score"]

        if not all(column in frame.columns for column in required):
            raise ValueError(
                f"Unexpected TSV columns: {list(frame.columns)}"
            )

        frame = frame[required]
    else:
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["query_id", "corpus_id", "score"],
            dtype=str,
        )

    qrels = {}

    for row in frame.itertuples(index=False):
        query_id = str(row.query_id).strip()
        document_id = str(row.corpus_id).strip()

        if not query_id or not document_id:
            continue

        try:
            score = int(row.score)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid relevance score {row.score!r} "
                f"in {path.name}"
            ) from error

        qrels.setdefault(query_id, {})[document_id] = score

    return qrels


def load_scifact(split: str = "test") -> tuple[dict, dict, dict]:
    """
    Load the SciFact corpus, queries, and relevance judgments.

    Args:
        split: Either 'train' or 'test'.

    Returns:
        corpus: Dictionary of scientific documents.
        queries: Dictionary of scientific claims.
        qrels: Query IDs mapped to relevant document IDs and scores.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"SciFact directory does not exist: {DATASET_DIR}"
        )

    corpus_path = find_dataset_file("corpus.jsonl")
    queries_path = find_dataset_file("queries.jsonl")
    qrels_path = find_dataset_file(f"{split}.tsv")

    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    qrels = load_qrels(qrels_path)

    if not corpus:
        raise ValueError("The SciFact corpus is empty.")

    if not queries:
        raise ValueError("The SciFact queries are empty.")

    print(f"Loaded {len(corpus)} scientific documents.")
    print(f"Loaded {len(queries)} queries.")
    print(f"Loaded relevance judgments for {len(qrels)} queries.")

    return corpus, queries, qrels
