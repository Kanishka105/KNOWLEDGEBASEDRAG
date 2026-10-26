1. RAG: The system will let a user ask questions about a collection of documents. It will retrieve relevant evidence, use an LLM to formulate an answer, and show where the answer came from.
2. Project: SciFact is a scientific claim-verification and evidence-retrieval benchmark. It was created to study whether systems can find scientific papers that support or refute scientific claims. In the BEIR format, it provides documents, queries, and relevance judgments
3. PHASE 1
a.Can our system find the right scientific papers in the first place?
SciFact Dataset -- Dataset Loader and Inspection -- BM25 baseline -- Ranked Scientific Papers
# Folder or file	Why it exists
datasets/scifact/	
Stores the original scientific dataset locally so we don't need to download it every time.
corpus.jsonl
Contains scientific paper records, including document IDs, titles, and text.
queries.jsonl	
Contains scientific claims that we want to search for.
qrels/test.tsv	
Tells us which documents are relevant to particular test queries. Used to evaluate retrieval quality.
qrels/train.tsv	
Provides training-split relevance judgments for experiments that use the training split.
download_dataset.py	
Downloads and extracts the dataset.
phase1_dataset/	
Groups all code responsible for preparing and experimenting with the dataset.
__init__.py	
Marks the folder as a Python package, making imports and module execution straightforward.
dataset_loader.py	
Loads the dataset into Python dictionaries.
inspect_dataset.py	
Lets us inspect sample records and verify that IDs and relevance judgments line up.
bm25_baseline.py	
Builds the first search index and returns ranked papers for a query.
