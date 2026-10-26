# Project: Scientific Research Assistance
It find the relevant papers, understands the evidence and generate answer and evaluate how reliable its results are

# Our solution: Build a system that retrieves relevant scientific documents, selects the strongest evidence, and uses an LLM to generate an answer grounded in those documents.

# RAG: Retrieval: find relevant info from external knowledge src
Generation: use language model to produce natural language answer informed by the retrieved info.
# Topics:
a. BM25[Keyword Retrieval]:It is an algo that ranks the doc according to how relevant they are to search query.It assign the score to each doc based on its query term matches and rank from highest to lowest score.
 1. Term Frequency(TF): how often a query term appears in a doxument but used freq saturations so repeating a term does not keep inc its contributions linearly.
 2. Inverse Doc Frequency: measure how info is a term across the whole doc collection. So BM25 give rarer terms more weight.
 3. Doc length Normalization: same query terms with same freq  then longer doc can gain an adv simply because they contain more text. 
b. Vector Query[Semantic Search]:A vector query is a search query represented as a numerical vector (embedding) that is used to find documents with similar meanings.An embedding is a numerical representation of text produced by an embedding model.
c. Vector Database: Store and retrieve embeddings
Stores document embeddings and associated text or metadata, and supports similarity searches over those vectors.

# Project:
Phase 1: Dataset foundation and baseline
Retrieval -- Augmentation: add info to prompt given to the LLM -- Generation: LLM produce an answer
1. Topics: Knowledge Corpus: collection of info[PDF,Reports,ect] available to rag.
2. Queries and relvance Judgements: A query is the research question or claim used to search the corpus.A relevance judgment identifies which documents the benchmark considers relevant to that query.
Tokenization: Converts text into the individual tokens so a retrieval algo can process it.
Inverted Indexes: maps terms to the doc that contain them.
# Purpose in Phase 1: Establish a keyword-search baseline before introducing embedding-based retrieval.
# A benchmark is a standardized dataset and evaluation procedure used to compare methods.We will evaluate the top five retrieved documents with:- Precision@5: fraction of the top five that are relevant.- Recall@5: fraction of all known relevant documents found in the top five.- MRR: how highly the first relevant document is ranked.Purpose in Phase 1: Quantify retrieval performance so later improvements can be measured.