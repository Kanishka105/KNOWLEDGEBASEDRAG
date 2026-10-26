# Each file do
embedding_model.py	
Load a pretrained sentence-embedding model
document_embedder.py	
Convert scientific papers into vectors
vector_store.py	
Store vectors and perform similarity search
semantic_search.py	
Embed a query and retrieve the top five papers
evaluate_dense.py	
Evaluate dense retrieval using the same test queries and metrics as Phase 1
README.md	
Document the model, settings, and results

# Semantic Retrieval:
Dataset-- Normalize text -- Embedding model -- Vector Index -- Similarity search -- evaluate -- save result.

Why we need Phase 2:
BM25: 5 mainly rewards overlapping terms. It may miss relevant papers when the vocabulary differs.
Dense Retrieval: converts the query and each paper into numerical vectors that represent semantic information learned by a model.
# The retrieval system compares those vectors and ranks papers by similarity.


TECH:
1. Sentence Transformers
A Python library that converts sentnces and documents into numerical vectors called embeddings.
Think: the embedding toolkit.
2. all-MiniLM-L6-v2
A pretrained sentence-embedding model loaded through Sentence Transformers.
Think: the specific model doing the language understanding.
3. NumPy
A Python library for working with arrays, matrices, vectors, and mathematical operations.
Think: the numerical calculation tool.
4. FAISS
A library for efficiently searching large collections of vectors by similarity.

Think: the vector search engine.


# One distinction: FAISS is optional in the implementation we've built so far. Your current dense search uses NumPy dot products. You do not need FAISS to complete this version of Phase 3, but if you want FAISS as part of your required technology stack, we must implement and test it separately.