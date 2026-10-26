# Hybrid Retrieval Mode:
means exact keyword matches and semantic meaning
MAIN PURPOSE: Combines the rankings from BM25 and dense retrieval using Reciprocal Rank Fusion (RRF). The objective is to benefit from both keyword matching and semantic understanding

# Rank fusion: Recriprocal Rank Fusion[RRF] means combine rankings directly comparing incompatible raw scores.
We'll use Reciprocal Rank Fusion (RRF) as the initial fusion method. BM25 scores and cosine similarities have different scales, so combining their raw scores directly can be misleading. RRF combines the ranks instead. Has Formula

# Hybrid top k results

# files and their responsibilities:
File	Main responsibility
__init__.py	
Marks the directory as a Python package.
hybrid_retrieval.py	
Implements BM25 + dense retrieval + RRF fusion.
hybrid_search.py	
Runs a convenient sample query and displays results.
evaluate_hybrid.py
Evaluates the hybrid method against relevance judgments.
README.md	
Documents the phase, commands, algorithm, and outputs.

Retrieve candidates: BM25 and FAISS retrieve relevant candidate papers.
all-MiniLM-L6-v2, independently converts the query and each document into vectors. This is efficient for searching thousands of documents.
Rerank candidates: A CrossEncoder evaluates each query–document pair and reorders the candidates by relevance.
A CrossEncoder works differently. It receives the query and a candidate document together, then predicts a relevance score for that pair.
# File 1
1. hybrid_retrieval.py
a. build_hybrid_index(): loads embeddings and documents id and check IDS number match with embedding rows. Build BM25 index from normalized doc text. Keep list in exact order used by bm25
2. retrieve_hybrid()
executes both retrieval branches for each query and bm25 ranks doc based on lexical matching.Dense retrieval embeds the query and computes similarity against the stored vectors.
RRF combines the two ranked lists.
The function returns the highest-ranked documents, including their rank in each individual method.
3. run_sample_query()
loads the dataset and choose the judged query, invokes retrieval and prints the top 5 results and save them for JSON.

# NOTE
Component	Responsibility
BM25Okapi	Retrieves candidates using query keywords.
SentenceTransformer	Converts the query into an embedding.
all-MiniLM-L6-v2	Supplies the 384-dimensional semantic representation used by FAISS.
faiss.IndexFlatIP	Searches the cached document vectors using exact inner-product similarity.
reciprocal_rank_fusion()	Combines BM25 and FAISS rankings.
CrossEncoder	Scores the query against each shortlisted document and reranks them.
NumPy	Validates vectors, sorts rankings, and processes scores.
JSON output	Saves final rankings and model information for inspection.


TOPICS
1. Sparse retrieval searches text using lexical features, such as words and their importance in documents.
2. Dense retrieval represents text as a compact, fixed-size vector of learned numerical features.
Property	Sparse / BM25	Dense / embeddings
Matches exact terminology	Usually strong	Depends on the model
Handles paraphrases	Limited by lexical overlap	Often better
Interpretable matching signals	Relatively easy to inspect	Less directly interpretable
Main limitation	Can miss related wording	Can miss rare terms or precise identifiers
Your implementation	rank_bm25	Sentence Transformers + FAISS
# Neither method is always better, especially for scientific claims containing specialized terminology.

Candidate retrieval is the first stage of a multi-stage search system. Its job is to find a reasonably small set of potentially relevant documents from a much larger corpus.
Your SciFact corpus contains around 5,183 documents. Instead of applying the expensive CrossEncoder to every document, your pipeline first retrieves candidates using BM25 and FAISS.
BM25 returns up to 100 keyword matches.
FAISS returns up to 100 semantic matches.
RRF combines these ranked lists.
The combined candidate set can contain at most 200 unique documents, and often fewer because the lists overlap.
The CrossEncoder reranks the top 50 fused candidates.

# Why not rerank all documents?
CrossEncoder scoring requires processing each query-document pair together. Scoring thousands of documents for every query is more expensive than searching vectors and using BM25 to shortlist candidates.
Candidate retrieval reduces that cost, but it introduces an important limitation: if a relevant document never enters the candidate set, the reranker cannot recover it.
That is why candidate retrieval quality is critical.

Rank fusion combines the rankings from multiple retrieval methods into one ranking.

# Reranking takes an existing candidate list and reorders its documents using a more detailed relevance model.

In your project, the reranker is:

cross-encoder/ms-marco-MiniLM-L-6-v2


# Summary
Concept	Main question it answers	Your implementation
Sparse retrieval	Which documents match the query's words?	BM25
Dense retrieval	Which documents have similar learned representations?	all-MiniLM-L6-v2 + FAISS
Candidate retrieval	Which documents should proceed to the next stage?	Top candidates from BM25 and FAISS
Rank fusion	How do we combine two ranked lists?	RRF
Reranking	Which shortlisted documents appear most relevant to this query?	CrossEncoder
Precision-recall trade-off	How many retrieved documents are relevant, and how many relevant documents did we find?	Evaluation metrics
Retrieval pipeline	How do all the stages work together?	BM25 → FAISS → RRF → CrossEncoder