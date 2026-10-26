from app.ai.rag.phase3_hybrid_retrieval.hybrid_retrieval import (
    reciprocal_rank_fusion,
)

# Pretend these are the rankings returned by two search methods.
bm25_results = ["DOC_A", "DOC_B", "DOC_C"]
dense_results = ["DOC_B", "DOC_C", "DOC_A"]

fused = reciprocal_rank_fusion(
    [bm25_results, dense_results],
    k=60,
)

ranked = sorted(
    fused,
    key=lambda doc_id: fused[doc_id],
    reverse=True,
)

print("BM25 ranking:", bm25_results)
print("Dense ranking:", dense_results)
print("\nFused ranking:")

for rank, doc_id in enumerate(ranked, start=1):
    print(f"{rank}. {doc_id}: {fused[doc_id]:.6f}")

assert ranked[0] == "DOC_B"
assert len(fused) == 3

print("\nRRF test passed!")