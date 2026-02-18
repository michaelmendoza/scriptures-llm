import re
from pathlib import Path

import chromadb
import ollama

from rag import config


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function that uses Ollama."""

    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            response = ollama.embed(model=self.model, input=text)
            embeddings.append(response["embeddings"][0])
        return embeddings


def get_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=OllamaEmbeddingFunction(),
    )


def load_documents(data_dir: Path = config.DATA_DIR) -> list[dict]:
    """Load all .txt and .md files from the data directory."""
    documents = []
    for ext in ("*.txt", "*.md"):
        for filepath in data_dir.rglob(ext):
            text = filepath.read_text(encoding="utf-8")
            if text.strip():
                documents.append({
                    "source": str(filepath.relative_to(data_dir)),
                    "text": text,
                })
    return documents


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            # Build overlap from the end of the current chunk
            words = current_chunk.split()
            overlap_text = ""
            for word in reversed(words):
                candidate = word + " " + overlap_text if overlap_text else word
                if len(candidate) > overlap:
                    break
                overlap_text = candidate
            current_chunk = overlap_text + " " + sentence if overlap_text else sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def ingest_documents():
    """Full ingestion pipeline: load, chunk, embed, and store documents."""
    documents = load_documents()
    if not documents:
        print(f"No .txt or .md files found in {config.DATA_DIR}/")
        return

    collection = get_collection()

    # Clear existing data for a fresh ingest
    existing = collection.count()
    if existing > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
        print(f"Cleared {existing} existing chunks.")

    total_chunks = 0
    all_texts = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}::chunk_{i}"
            all_texts.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": doc["source"],
                "chunk_index": i,
            })
        total_chunks += len(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    # Add all at once (ChromaDB handles batching internally)
    if all_texts:
        collection.add(
            documents=all_texts,
            ids=all_ids,
            metadatas=all_metadatas,
        )

    print(f"\nIngested {len(documents)} files -> {total_chunks} chunks")


if __name__ == "__main__":
    ingest_documents()
