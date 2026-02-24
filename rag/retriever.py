"""Retrieval module: vector search, BM25 hybrid search, and reranking."""

import re

from rag import config
from rag.ingest import get_collection

# Module-level cache for BM25 index
_bm25_index = None
_bm25_corpus_ids: list[str] = []
_bm25_corpus_docs: list[dict] = []


def _vector_search(query: str, top_k: int) -> list[dict]:
    """Run a vector similarity search against ChromaDB."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i] if results["distances"] else None,
        })

    return chunks


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


def _get_bm25_index():
    """Lazily build and cache a BM25 index from all documents in ChromaDB."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs

    if _bm25_index is not None:
        return _bm25_index

    from rank_bm25 import BM25Okapi

    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_docs = []
    tokenized_corpus = []

    for i, doc_id in enumerate(all_data["ids"]):
        doc_text = all_data["documents"][i]
        doc_meta = all_data["metadatas"][i]
        _bm25_corpus_docs.append({
            "id": doc_id,
            "text": doc_text,
            "source": doc_meta.get("source", ""),
            "metadata": doc_meta,
        })
        tokenized_corpus.append(_tokenize(doc_text))

    _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index


def bm25_search(query: str, top_k: int) -> list[dict]:
    """Run a BM25 keyword search over the cached corpus."""
    index = _get_bm25_index()
    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # Get top-k indices
    scored_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in scored_indices:
        entry = dict(_bm25_corpus_docs[idx])
        entry["score"] = float(scores[idx])
        results.append(entry)

    return results


def invalidate_bm25_cache():
    """Clear the BM25 cache (call after re-ingesting)."""
    global _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs
    _bm25_index = None
    _bm25_corpus_ids = []
    _bm25_corpus_docs = []


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a query.

    Supports hybrid search (vector + BM25 with RRF fusion) and cross-encoder
    reranking when the respective feature flags are enabled.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Determine how many candidates to fetch
    fetch_k = config.RERANK_INITIAL_K if config.ENABLE_RERANKING else top_k

    if config.ENABLE_HYBRID_SEARCH:
        # Run both vector and BM25 searches
        vector_results = _vector_search(query, fetch_k)
        bm25_results = bm25_search(query, fetch_k)

        # Merge with Reciprocal Rank Fusion
        from rag.reranker import reciprocal_rank_fusion
        candidates = reciprocal_rank_fusion(vector_results, bm25_results)
    else:
        candidates = _vector_search(query, fetch_k)

    # Optionally rerank with cross-encoder
    if config.ENABLE_RERANKING and candidates:
        from rag.reranker import cross_encoder_rerank
        candidates = cross_encoder_rerank(query, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    # Normalize output format (ensure backward-compatible keys)
    results = []
    for c in candidates:
        results.append({
            "text": c["text"],
            "source": c.get("source", ""),
            "score": c.get("rerank_score") or c.get("rrf_score") or c.get("score"),
            "metadata": c.get("metadata", {}),
        })

    return results
