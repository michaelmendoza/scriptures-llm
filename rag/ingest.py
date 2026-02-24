"""Document ingestion pipeline: load, chunk, enrich, embed, and store."""

import json
import re
from pathlib import Path

import chromadb
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rag import config
from rag.chunking import chunk_text as _chunk_text_dispatch, chunk_sentences

console = Console()


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


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Legacy chunking (kept for backward-compatibility when flags are off)
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    return chunk_sentences(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Metadata enrichment helpers
# ---------------------------------------------------------------------------

def _load_metadata() -> dict:
    """Build a metadata lookup from ``data/metadata/``.

    Returns ``{book_slug: {volume, book_title, chapters, verses,
    book_summary, chapter_summaries: {int: str}, ...}}``
    """
    lookup: dict[str, dict] = {}
    meta_dir = config.METADATA_DIR

    for vol_meta_path in sorted(meta_dir.glob("*/metadata.json")):
        vol_meta = json.loads(vol_meta_path.read_text(encoding="utf-8"))
        volume_title = vol_meta["volume_title"]
        volume_slug = vol_meta["slug"]

        for book in vol_meta.get("books", []):
            book_slug = book["slug"]
            entry: dict = {
                "volume": volume_title,
                "volume_slug": volume_slug,
                "book_title": book["title"],
                "book_slug": book_slug,
                "total_chapters": book["chapters"],
                "total_verses": book.get("verses", 0),
                "book_summary": "",
                "chapter_summaries": {},
            }

            # Try to load book-level metadata (LLM-generated)
            book_meta_path = meta_dir / volume_slug / f"{book_slug}.meta.json"
            if book_meta_path.exists():
                book_meta = json.loads(book_meta_path.read_text(encoding="utf-8"))
                entry["book_summary"] = book_meta.get("summary", "")
                for cs in book_meta.get("chapter_summaries", []):
                    entry["chapter_summaries"][cs["chapter"]] = cs["summary"]

            # Try to load individual chapter summaries (fallback)
            chapters_dir = meta_dir / volume_slug / "chapters"
            if chapters_dir.is_dir():
                for ch_path in chapters_dir.glob(f"{book_slug}-*.summary.json"):
                    ch_data = json.loads(ch_path.read_text(encoding="utf-8"))
                    ch_num = ch_data.get("chapter")
                    if ch_num and ch_num not in entry["chapter_summaries"]:
                        entry["chapter_summaries"][ch_num] = ch_data.get("summary", "")

            lookup[book_slug] = entry

    return lookup


def _find_section_heading(text: str, chunk_start: int) -> str:
    """Find the nearest markdown heading above *chunk_start* in *text*."""
    before = text[:chunk_start]
    headings = re.findall(r'^(#{1,6}\s+.+)$', before, re.MULTILINE)
    return headings[-1].strip() if headings else ""


def _parse_source(source: str) -> tuple[str, str, int | None]:
    """Extract (volume_slug, book_slug, chapter_num) from a source path.

    Examples:
        ``old-testament/genesis-1.md`` → ``("old-testament", "genesis", 1)``
        ``new-testament/1-corinthians-3.md`` → ``("new-testament", "1-corinthians", 3)``
    """
    path = Path(source)
    volume_slug = path.parent.name  # e.g. "old-testament"
    stem = path.stem                # e.g. "genesis-1" or "1-corinthians-3"

    # Split off the trailing chapter number
    match = re.match(r'^(.+)-(\d+)$', stem)
    if match:
        book_slug = match.group(1)
        chapter_num = int(match.group(2))
    else:
        book_slug = stem
        chapter_num = None

    return volume_slug, book_slug, chapter_num


def extract_metadata(
    source: str,
    chunk_index: int,
    total_chunks: int,
    text: str,
    full_text: str,
    metadata_lookup: dict,
) -> dict:
    """Build enriched metadata for a single chunk."""
    meta: dict = {
        "source": source,
        "chunk_index": chunk_index,
    }

    volume_slug, book_slug, chapter_num = _parse_source(source)
    meta["volume"] = volume_slug
    meta["book_slug"] = book_slug
    if chapter_num is not None:
        meta["chapter"] = chapter_num

    # Enrich from metadata lookup
    book_info = metadata_lookup.get(book_slug, {})
    if book_info:
        meta["book"] = book_info.get("book_title", "")
        if book_info.get("book_summary"):
            meta["book_summary"] = book_info["book_summary"]
        if chapter_num and chapter_num in book_info.get("chapter_summaries", {}):
            meta["chapter_summary"] = book_info["chapter_summaries"][chapter_num]

    # Document title (first heading)
    title_match = re.match(r'^#\s+(.+)$', full_text, re.MULTILINE)
    if title_match:
        meta["document_title"] = title_match.group(1).strip()

    # Section heading
    chunk_start = full_text.find(text[:80]) if text else 0
    if chunk_start > 0:
        heading = _find_section_heading(full_text, chunk_start)
        if heading:
            meta["section_heading"] = heading

    # Position
    meta["total_chunks"] = total_chunks
    if total_chunks > 1:
        if chunk_index == 0:
            meta["position"] = "start"
        elif chunk_index == total_chunks - 1:
            meta["position"] = "end"
        else:
            meta["position"] = "middle"
    else:
        meta["position"] = "only"

    return meta


# ---------------------------------------------------------------------------
# Contextual retrieval (LLM-generated chunk prefixes)
# ---------------------------------------------------------------------------

def generate_chunk_context(chunk_text_str: str, full_text: str) -> str:
    """Generate a 2-3 sentence contextual prefix for a chunk via LLM.

    The prefix situates the chunk within its parent document to improve
    retrieval relevance.
    """
    # Send first 3000 chars of the parent document for context
    doc_context = full_text[:3000]

    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document-context generator. Given a chunk of text and its "
                    "parent document, write 2-3 concise sentences that situate the chunk "
                    "within the broader document. Focus on what this specific passage is "
                    "about and how it relates to the document as a whole. "
                    "Do NOT summarize the full document. Only provide the context prefix."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PARENT DOCUMENT (first 3000 chars):\n{doc_context}\n\n"
                    f"---\n\nCHUNK:\n{chunk_text_str}"
                ),
            },
        ],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

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

    # Load metadata if enrichment is enabled
    metadata_lookup: dict = {}
    if config.ENABLE_METADATA_ENRICHMENT:
        console.print("[dim]Loading metadata for enrichment...[/dim]")
        metadata_lookup = _load_metadata()
        console.print(f"[dim]  Loaded metadata for {len(metadata_lookup)} book(s)[/dim]")

    # Use dispatcher when advanced chunking is enabled, otherwise legacy
    use_advanced_chunking = (
        config.ENABLE_CONTEXT_AWARE_CHUNKING or config.ENABLE_SEMANTIC_CHUNKING
    )
    chunk_fn = _chunk_text_dispatch if use_advanced_chunking else chunk_text

    total_chunks = 0
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc in documents:
        chunks = chunk_fn(doc["text"])
        num_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}::chunk_{i}"

            # Optionally generate contextual prefix
            if config.ENABLE_CONTEXTUAL_RETRIEVAL:
                prefix = generate_chunk_context(chunk, doc["text"])
                chunk = f"{prefix}\n\n{chunk}"

            all_texts.append(chunk)
            all_ids.append(chunk_id)

            # Build metadata
            if config.ENABLE_METADATA_ENRICHMENT and metadata_lookup:
                meta = extract_metadata(
                    source=doc["source"],
                    chunk_index=i,
                    total_chunks=num_chunks,
                    text=chunk,
                    full_text=doc["text"],
                    metadata_lookup=metadata_lookup,
                )
            else:
                meta = {
                    "source": doc["source"],
                    "chunk_index": i,
                }

            all_metadatas.append(meta)

        total_chunks += num_chunks
        print(f"  {doc['source']}: {num_chunks} chunks")

    print(f'Pre-processed {len(documents)} files into {total_chunks} chunks')

    # Add in batches to stay under ChromaDB's max batch size
    batch_size = 5000
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding & storing", total=len(all_texts))
        for start in range(0, len(all_texts), batch_size):
            end = start + batch_size
            collection.add(
                documents=all_texts[start:end],
                ids=all_ids[start:end],
                metadatas=all_metadatas[start:end],
            )
            progress.advance(task, advance=min(batch_size, len(all_texts) - start))

    print(f"\nIngested documents into ChromaDB collection.")


if __name__ == "__main__":
    ingest_documents()
