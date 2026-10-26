# Keyword Retrieval Phase
Dataset -- Inspect -- Normalize text -- tokenize -- BM25 -- Evaluate -- Save -- Baseline
1. Data Loader: to load the data carefully
2. Inspect Data: To load -- inspect -- find -- display
3. The code below assumes these fields:
- Corpus: doc_id, title, abstract or text.
- Queries: query_id, text.
- Relevance judgments: query_id, doc_id.
# dataset_loader.py
How this file works
- load_jsonl() reads each JSONL record.
- normalize_document_text() combines a paper's title and available abstract or text.
- load_corpus() preserves the document IDs and searchable text.
- load_queries() loads query IDs and query text.
- load_qrels() converts relevance judgments into a dictionary mapping query IDs to sets of relevant document IDs.
- load_dataset() loads everything through a single function.
# inspect_dataset.py
Before implementing retrieval, you should verify that your data is loaded correctly.
This script checks:
- Number of papers and queries.
- Number of queries with relevance judgments.
- Missing or empty document text.
- Example titles, abstracts, queries, and relevance labels.
- Whether query and document IDs match the loaded dataset.
# bm25_baseline.py
For every evaluation query:
1. Convert the query into tokens.
2. Calculate BM25 scores for the corpus.
3. Sort the documents by score in descending order.
4. Select the top five documents.
5. Compare the retrieved IDs with the known relevant document IDs.
6. Calculate evaluation metrics.
7. Save the results.

unction	Responsibility
tokenize()	Converts text into lowercase tokens.
build_bm25()	Builds the BM25 model from the corpus.
retrieve()	Scores and ranks documents for a query.
precision_at_k()	Measures the proportion of the top five that are relevant.
recall_at_k()	Measures how many known relevant documents appear in the top five.
reciprocal_rank()	Measures the position of the first relevant result.
evaluate_query()	Combines retrieval results with relevance judgments.
main()	Runs the complete experiment and saves its outputs.


# Evaluation
Metric	Interpretation
Precision@5 = 0.158	On average, about 0.79 of the five retrieved documents are relevant per query.
Recall@5 = 0.7284	The top five retrieve about 72.84% of the relevant documents, averaged across queries under your current evaluation definition.
MRR@5 = 0.6132	Rewards finding the first relevant paper near the top of the ranking.
nDCG@5 = 0.6360	Measures how well relevant papers are ranked near the top.

# A benchmark dataset is a standard, prepared collection of data used to test and compare how well different models or algorithms perform on the same task.
Suppose you build three search systems for scientific research papers.
- Phase 1: BM25 keyword search
- Phase 2: Dense semantic search
- Phase 3: Hybrid retrieval
You need a common dataset to check which system retrieves the most relevant research papers. That dataset acts as your benchmark.
# In short, the benchmark dataset provides the test data and reference relevance judgments; your retrieval algorithms compete to retrieve the right documents.