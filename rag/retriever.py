from rag import config
from rag.ingest import get_collection


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
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
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": results["distances"][0][i] if results["distances"] else None,
        })

    return chunks
