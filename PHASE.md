Phase 1 — Dataset and baseline
Current starting phase


Overview: Understand SciFact and establish a simple search baseline.
Tasks
- Inspect corpus.jsonl, queries.jsonl, and qrels/test.tsv.
- Load documents, queries, and relevance judgments into Python.
- Examine document IDs, titles, text, query IDs, and relevance labels.
- Build a basic keyword-search baseline using BM25.
Why: You need a reference system to compare against your later RAG improvements.
Technologies: Python, BEIR, JSONL, pandas, rank-bm25 or a BM25 implementation.
Deliverable: A query returns a ranked list of scientific papers, with baseline retrieval metrics.


Phase 2 — Document ingestion and chunking

Overview: Prepare research documents for passage-level search.
Tasks
- Load scientific paper text and metadata.
- Clean malformed or empty text without removing important scientific terms.
- Split long documents into chunks, with configurable size and overlap.
- Preserve the original document ID, title, and chunk identifier.
Why: Relevant evidence may occupy only a small part of a paper. Chunking helps the system retrieve that evidence directly.
Technologies: Python, custom chunker, tokenizer utilities, PyMuPDF for future PDF uploads.
Deliverable: A collection of traceable passages ready for indexing.


Phase 3 — Embeddings and vector database

Overview: Convert passages into numerical representations and create a searchable index.
Tasks
- Load a pretrained embedding model.
- Generate embeddings for each passage.
- Store vectors and passage metadata in FAISS.
- Save and reload the index for repeatable experiments.
Why: Vector representations let us search for semantic similarity rather than relying only on matching words.
Technologies: Sentence Transformers, all-MiniLM-L6-v2, NumPy, FAISS.
Deliverable: A reusable vector index that can return similar passages for a query.


Phase 4 — Semantic retrieval

Overview: Build the first embedding-based scientific search engine.
Tasks
- Embed each user query.
- Search FAISS for the nearest passage vectors.
- Return the top-k passages with similarity scores.
- Map each result back to its original scientific paper.
Why: This is the first stage that finds relevant scientific evidence based on meaning.
Technologies: Sentence Transformers, FAISS, NumPy.
Deliverable: A working semantic retriever and ranked research evidence.


Phase 5 — Advanced retrieval

Overview: Improve the quality of retrieved documents.
Tasks
- Add BM25 keyword retrieval alongside vector search.
- Combine results using a method such as Reciprocal Rank Fusion (RRF).
- Add metadata filters and query rewriting where useful.
- Optionally rerank candidates with a cross-encoder.
Why: Semantic search can miss exact scientific terminology, while keyword search can miss meaning expressed with different wording. Combining them can improve retrieval.
Technologies: BM25, Sentence Transformers, RRF, cross-encoder reranking; Qdrant can replace FAISS later if needed.
Deliverable: Comparable keyword, semantic, hybrid, and reranked search methods.


Phase 6 — RAG generation and citations

Overview: Turn retrieved evidence into a readable scientific answer.
Tasks
- Construct a prompt containing the query and selected evidence.
- Call an LLM to synthesize an answer.
- Include source titles, document IDs, and relevant passages.
- Instruct the model to distinguish evidence from inference and abstain when evidence is insufficient.
Why: Retrieval finds documents; generation explains what those documents say.
Technologies: An LLM API or compatible local model, prompt templates, Python.
Deliverable: An end-to-end RAG pipeline that returns answers with traceable evidence.


Phase 7 — Evaluation and experiments

Overview: Measure whether your system retrieves and answers correctly.
Tasks
- Evaluate retrieval against SciFact qrels.
- Compare BM25, semantic, hybrid, and reranked retrieval.
- Test chunk sizes, embedding models, and top-k settings.
- Evaluate answer relevance, faithfulness, and citation support separately.
Why: A major project needs measurable evidence of improvement, not just a few good-looking responses.
Technologies: BEIR evaluation utilities, Recall@k, Precision@k, MRR, nDCG, Ragas or custom answer-evaluation tests.
Deliverable: Metrics, comparison tables, error analysis, and reproducible results.


Phase 8 — Conversational RAG

Overview: Let users conduct a multi-turn research session.
Tasks
- Maintain conversation history.
- Resolve follow-up questions using relevant context.
- Keep retrieved evidence tied to each answer.
- Prevent irrelevant old conversation context from contaminating new searches.
Why: Researchers often ask follow-up questions rather than starting a new search every time.
Technologies: Python, FastAPI, conversation storage, query rewriting, optional LangChain or LangGraph.
Deliverable: A conversational scientific research assistant.


Phase 9 — Frontend and API

Overview: Make the search engine usable through a web application.
Tasks
- Create a query box and ranked paper-results view.
- Display generated answers and clickable source references.
- Add document details, conversation history, and loading/error states.
- Expose the pipeline through API endpoints.
Why: A working interface makes the research system demonstrable to users and evaluators.
Technologies: React, Tailwind CSS, FastAPI, Pydantic.
Deliverable: A functional research assistant web application.


Phase 10 — Production readiness and deployment

Overview: Package, deploy, and maintain the system.
Tasks
- Add tests, validation, logging, and error handling.
- Add authentication and document access permissions if users upload private files.
- Containerize the application and configure deployment.
- Monitor latency, retrieval failures, and LLM/API costs.
- Document setup, architecture, limitations, and experimental findings.
Why: A local prototype is not the same as a reliable deployed application.
Technologies: Docker, Git, FastAPI, cloud hosting, automated tests, monitoring.
Deliverable: A deployable, documented major project.