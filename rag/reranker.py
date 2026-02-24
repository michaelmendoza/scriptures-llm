"""Reranking utilities: Reciprocal Rank Fusion and cross-encoder reranking."""

from rag import config

# Lazy-loaded cross-encoder model
_cross_encoder = None


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = config.RRF_K) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each list is a sequence of dicts that must contain an ``"id"`` key.
    Returns a de-duplicated list sorted by descending RRF score, with
    ``rrf_score`` added to each dict.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            # Keep the first occurrence (preserve metadata)
            if item_id not in items:
                items[item_id] = item

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results = []
    for item_id in sorted_ids:
        entry = dict(items[item_id])
        entry["rrf_score"] = scores[item_id]
        results.append(entry)

    return results


def cross_encoder_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = config.TOP_K,
) -> list[dict]:
    """Re-score candidates using a cross-encoder model.

    Lazily loads the ``CrossEncoder`` from ``sentence-transformers`` on first
    call.  Each candidate must have a ``"text"`` key.  Returns the top-*k*
    candidates sorted by descending cross-encoder score, with
    ``rerank_score`` added to each dict.
    """
    global _cross_encoder

    if not candidates:
        return []

    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)

    pairs = [(query, c["text"]) for c in candidates]
    scores = _cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort descending by rerank score and take top-k
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
