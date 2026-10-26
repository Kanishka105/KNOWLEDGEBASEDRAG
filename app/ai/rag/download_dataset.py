
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "datasets"

DATASET_NAME = "scifact"
DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)


def main():
    print("Downloading SciFact...", flush=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    data_path = util.download_and_unzip(
        DATASET_URL,
        str(DATASET_DIR),
    )

    print("Loading SciFact dataset...", flush=True)

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")

    print("\n--- Dataset loaded successfully ---")
    print("Documents:", len(corpus))
    print("Queries:", len(queries))
    print("Query groups with relevance judgments:", len(qrels))

    doc_id = next(iter(corpus))
    print("\nSample document ID:", doc_id)
    print(corpus[doc_id])

    query_id = next(iter(queries))
    print("\nSample query ID:", query_id)
    print(queries[query_id])


if __name__ == "__main__":
    main()
