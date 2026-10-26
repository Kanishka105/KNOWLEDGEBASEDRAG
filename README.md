# KNOWLEDGE-AI 🔎
### An Evidence-Grounded Scientific Research Assistant Powered by Retrieval-Augmented Generation

**KNOWLEDGE-AI** is an intelligent research assistant designed to make scientific knowledge easier to discover, retrieve, and understand. It combines traditional information retrieval, semantic search, hybrid ranking, and Large Language Models (LLMs) to retrieve relevant scientific evidence and generate context-aware answers.

Built with **Python, FastAPI, FAISS, Sentence Transformers, and Google Gemini**, the project explores how Retrieval-Augmented Generation (RAG) can improve access to scientific literature by grounding generated answers in retrieved documents rather than relying solely on an LLM's internal knowledge.

---

## 🎯 Project Overview

Scientific research is growing rapidly, making it increasingly difficult to find relevant papers, identify supporting evidence, and synthesize information efficiently. Traditional keyword search can miss semantically related documents, while standalone LLMs may generate answers that lack verifiable supporting evidence.

KNOWLEDGE-AI addresses these challenges through a modular RAG pipeline that combines keyword matching, semantic retrieval, hybrid ranking, and evidence-grounded answer generation.

### Project Objectives

- Retrieve relevant scientific documents using both lexical and semantic search.
- Combine complementary retrieval methods to improve document discovery.
- Rerank candidate documents to prioritize the most relevant evidence.
- Generate context-aware answers grounded in retrieved research.
- Support question answering over uploaded PDF documents.
- Evaluate retrieval quality using established information-retrieval metrics.
- Provide API endpoints for research, document search, and conversational workflows.

## ✨ Key Features

**1. Hybrid Information Retrieval**

Combines BM25 keyword retrieval and FAISS-based dense vector search using Reciprocal Rank Fusion (RRF), bringing together exact-term matching and semantic similarity.

**2. Semantic Search**

Uses Sentence Transformers to encode documents and queries into dense vectors, enabling retrieval based on meaning rather than exact keyword overlap alone.

**3. CrossEncoder Reranking**

Applies a CrossEncoder to score query-document pairs and refine the ranking of retrieved candidates.

**4. Grounded Question Answering**

Uses retrieved research evidence as context for Gemini to generate relevant answers. Source references can help users inspect the supporting documents.

**5. PDF-Based Research**

Supports uploading research documents, indexing their content, searching for relevant information, and asking questions about uploaded material.

**6. Conversational Research**

Provides conversation-related API functionality to support contextual interactions and follow-up questions.

**7. Agentic Research Workflow**

Includes an agentic research endpoint for a multi-step research workflow built around the project's retrieval and answer-generation capabilities.

**8. Retrieval Evaluation**

Includes an evaluation workflow for comparing retrieval approaches using Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG).

---

## 🏗️ System Architecture

The primary research-question workflow follows this architecture:

```text
                 USER QUERY
                     │
                     ▼
             QUERY PROCESSING
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
     BM25 RETRIEVAL       FAISS RETRIEVAL
     Keyword Matching     Semantic Search
           │                   │
           └─────────┬─────────┘
                     ▼
          RECIPROCAL RANK FUSION
                     │
                     ▼
          CROSSENCODER RERANKING
                     │
                     ▼
           TOP RELEVANT DOCUMENTS
                     │
                     ▼
           EVIDENCE + USER QUERY
                     │
                     ▼
             GEMINI GENERATION
                     │
                     ▼
          ANSWER WITH SOURCE CONTEXT
```

*Note: This diagram represents the intended retrieval and generation flow. The exact behavior depends on the configured pipeline and implementation.*

## 🔬 Retrieval Pipeline

The project is organized into modular phases, each addressing a different part of the RAG workflow.

### Phase 1 — Dataset Foundation and BM25 Baseline

Establishes the scientific document collection, research queries, and relevance judgments. BM25 provides a keyword-based baseline against which more advanced retrieval methods can be compared.

**Core concepts:** corpus preparation, tokenization, inverted-index retrieval, relevance judgments, and baseline evaluation.

### Phase 2 — Semantic Search with FAISS

Uses a Sentence Transformer model to generate dense vector representations of documents. FAISS indexes these embeddings and retrieves candidates based on vector similarity.

**Core concepts:** text embeddings, vector representations, similarity search, dense retrieval, and vector indexing.

### Phase 3 — Hybrid Retrieval

Combines BM25 and dense retrieval using Reciprocal Rank Fusion. RRF merges the rankings produced by different retrieval methods without requiring their raw scores to share the same scale.

**Core concepts:** sparse retrieval, dense retrieval, rank fusion, complementary search strategies, and candidate retrieval.

### Phase 4 — RAG Answer Generation

Uses retrieved documents as contextual evidence for answer generation through an LLM. This separates evidence retrieval from natural-language response generation.

**Core concepts:** retrieval-augmented generation, context construction, prompt design, grounded responses, and source attribution.

### Phase 5 — Retrieval Evaluation

Provides a framework for evaluating and comparing retrieval approaches on research queries with relevance judgments.

**Core concepts:** Precision@K, Recall@K, MRR, nDCG, comparative benchmarking, and retrieval error analysis.

### Phase 6 — Document Ingestion and PDF Question Answering

Extends the system beyond the initial scientific corpus by supporting uploaded-document workflows, allowing users to search and ask questions about their own research material.

**Core concepts:** document ingestion, text extraction, indexing, retrieval, and document-grounded question answering.

*Additional modules in the repository may extend this pipeline with advanced retrieval, incremental ingestion, or other experimental capabilities.*

---

## 🧠 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Core application and retrieval logic |
| API framework | FastAPI | Exposes research and document workflows |
| Data validation | Pydantic | Validates API request and response models |
| Keyword retrieval | BM25 | Retrieves documents using lexical relevance |
| Vector search | FAISS | Efficient dense-vector similarity search |
| Embeddings | Sentence Transformers | Encodes queries and documents |
| Rank fusion | Reciprocal Rank Fusion | Combines retrieval rankings |
| Reranking | CrossEncoder | Refines query-document relevance |
| Language model | Google Gemini API | Generates contextual answers |
| Research dataset | SciFact / BEIR | Supports scientific retrieval experiments |
| Frontend | React and Vite | Dashboard interface, subject to repository configuration |
| Testing | pytest | Automated test execution |

## 📂 Project Structure

The following illustrates the main logical modules. Consult the repository for the exact current filenames and additional components.

```text
KNOWLEDGEBASEDRAG/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── core/
│   │   └── conversation_store.py
│   │
│   └── ai/
│       └── rag/
│           ├── phase1_dataset/
│           ├── phase2_semantic_search/
│           ├── phase3_hybrid_retrieval/
│           ├── phase4_rag_generation/
│           ├── phase5_evaluation/
│           ├── phase6_document_ingestion/
│           └── ...
│
├── dashboard/
├── tests/
├── outputs/
├── requirements.txt
├── .env.example
├── README.md
├── PHASE.md
└── DETAIL.md
```

## 📊 Evaluation Strategy

A key objective is to compare retrieval methods under consistent experimental conditions.

| Metric | What it measures |
|---|---|
| Precision@K | Fraction of the top-K retrieved documents that are relevant |
| Recall@K | Fraction of all relevant documents retrieved in the top K |
| MRR | How highly the first relevant document is ranked |
| nDCG@K | Ranking quality, accounting for the positions of relevant results |

The planned comparison includes:

- **BM25:** lexical baseline.
- **FAISS dense retrieval:** semantic similarity baseline.
- **Hybrid RRF:** fusion of keyword and semantic rankings.
- **Hybrid + CrossEncoder:** reranking of hybrid-retrieved candidates.

For a fair comparison, all methods should use the same evaluation queries, relevance judgments, and cutoff values. Latency can also be recorded to understand the trade-off between retrieval quality and computational cost.

**Important:** Actual scores should be generated by running the evaluation scripts. Improvements should only be claimed when supported by measured results.

## 🚀 Getting Started

### Prerequisites

- Python and pip
- Git
- A Google Gemini API key for LLM-powered functionality
- The required dataset and vector-index artifacts
- Node.js and npm if running the frontend dashboard

### 1. Clone the Repository

```bash
git clone https://github.com/Kanishka105/KNOWLEDGEBASEDRAG.git
cd KNOWLEDGEBASEDRAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a local `.env` file based on `.env.example`. Add the required API credentials and configuration values using the variable names expected by the application.

**Security:** Never commit API keys, passwords, or other secrets to GitHub.

### 5. Prepare the Dataset and Indexes

Follow the dataset-loader and vector-store instructions in the relevant project modules. Retrieval workflows that depend on precomputed embeddings or FAISS indexes will require those artifacts to be generated or supplied.

### 6. Start the Backend

```bash
uvicorn app.api.main:app --reload
```

Open the interactive API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 7. Start the Frontend

If the dashboard has its own `package.json`, run:

```bash
cd dashboard
npm install
npm run dev
```

Use the scripts and configuration defined in the dashboard's `package.json`.

---

## 🔌 API Overview

The FastAPI application exposes endpoints for scientific question answering, document workflows, and conversations.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/api/v1/ask` | Ask a research question |
| POST | `/api/v1/agentic-research` | Run the agentic research workflow |
| POST | `/api/v1/documents/upload` | Upload a document |
| GET | `/api/v1/documents` | List available documents |
| POST | `/api/v1/documents/search` | Search document content |
| POST | `/api/v1/documents/ask` | Ask questions about documents |
| POST | `/api/v1/conversations` | Create a conversation |
| GET | `/api/v1/conversations` | List conversations |

Additional conversation endpoints support retrieving conversation details and adding messages. Consult `/docs` for request schemas, response formats, and the complete route list.

## 🧪 Running Tests and Evaluation

Run the automated test suite:

```bash
pytest -v
```

Run a small retrieval evaluation:

```bash
python -m app.ai.rag.phase5_evaluation.evaluate_retrieval --max-queries 10
```

Run all eligible evaluation queries:

```bash
python -m app.ai.rag.phase5_evaluation.evaluate_retrieval --max-queries 0
```

The evaluation script is configured to write results to:

```text
outputs/phase5/retrieval_metrics.json
```

Check the script's current options and dataset requirements before running these commands.

## ⚠️ Current Scope and Limitations

- Retrieval quality depends on the dataset, query, embedding model, and ranking configuration.
- Answers are only as reliable as the evidence retrieved and the model's interpretation of that evidence.
- Source references should be checked against the original documents.
- Dataset and vector-index artifacts may need separate preparation.
- Performance, scalability, and answer faithfulness require systematic evaluation before production use.

## 🛣️ Future Roadmap

- Complete reproducible benchmarks for all retrieval configurations.
- Improve retrieval quality through error analysis and parameter tuning.
- Evaluate answer faithfulness, citation correctness, and unsupported claims.
- Optimize indexing, retrieval latency, and document ingestion.
- Strengthen persistence, monitoring, and deployment workflows.
- Expand the dashboard and improve the research experience.

## 📚 Learning Outcomes

This project brings together practical concepts from information retrieval, Natural Language Processing, vector databases, hybrid search, reranking, RAG architecture, API development, and experimental evaluation.

It is designed to demonstrate how a research assistant can progress from a simple keyword-based baseline toward a more comprehensive evidence-grounded question-answering system.

## 📄 License

See the [`LICENSE`](LICENSE) file for the project's licensing terms.

---

**KNOWLEDGE-AI — Retrieve the evidence. Understand the research. Ground the answer.**
