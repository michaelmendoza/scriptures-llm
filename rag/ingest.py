"""Document ingestion pipeline: load, chunk, enrich, embed, and store."""

import hashlib
import json
import re
from pathlib import Path

import chromadb
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from rag import config
from rag.chunking import chunk_text as _chunk_text_dispatch, chunk_sentences

console = Console()


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function that uses Ollama."""

    def __init__(self, model: str = config.EMBEDDING_MODEL):
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = ollama.embed(model=self.model, input=input)
        return response["embeddings"]


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

def _content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest for *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _existing_source_hashes(collection) -> dict[str, str]:
    """Return ``{source: content_hash}`` for documents already in the collection."""
    count = collection.count()
    if count == 0:
        return {}
    data = collection.get(include=["metadatas"])
    hashes: dict[str, str] = {}
    for meta in data["metadatas"]:
        src = meta.get("source", "")
        h = meta.get("content_hash", "")
        if src and h:
            hashes[src] = h
    return hashes


def ingest_documents(fresh: bool = False):
    """Full ingestion pipeline: load, chunk, embed, and store documents.

    When *fresh* is ``True`` the collection is wiped first.  Otherwise only
    new or modified documents are ingested (incremental mode).
    """
    documents = load_documents()
    if not documents:
        console.print(f"[yellow]No .txt or .md files found in {config.DATA_DIR}/[/yellow]")
        return

    collection = get_collection()

    # Fresh mode: wipe collection
    if fresh:
        existing = collection.count()
        if existing > 0:
            all_ids = collection.get()["ids"]
            collection.delete(ids=all_ids)
            console.print(f"[dim]Cleared {existing} existing chunks (fresh ingest).[/dim]")
        existing_hashes: dict[str, str] = {}
    else:
        existing_hashes = _existing_source_hashes(collection)

    # Filter to new / modified documents
    docs_to_ingest: list[dict] = []
    skipped = 0
    for doc in documents:
        doc["content_hash"] = _content_hash(doc["text"])
        prev_hash = existing_hashes.get(doc["source"])
        if prev_hash and prev_hash == doc["content_hash"]:
            skipped += 1
            continue
        # If the source exists but content changed, delete old chunks first
        if prev_hash:
            old_ids = [
                cid for cid in collection.get()["ids"]
                if cid.startswith(f"{doc['source']}::")
            ]
            if old_ids:
                collection.delete(ids=old_ids)
        docs_to_ingest.append(doc)

    if skipped:
        console.print(f"[dim]Skipped {skipped} unchanged document(s).[/dim]")

    if not docs_to_ingest:
        console.print("[green]All documents already ingested. Nothing to do.[/green]")
        return

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

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    # --- Phase 1: Chunking documents ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking documents", total=len(docs_to_ingest))
        for doc in docs_to_ingest:
            chunks = chunk_fn(doc["text"])
            num_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['source']}::chunk_{i}"

                all_texts.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "_doc_source": doc["source"],
                    "_doc_text": doc["text"],
                    "_chunk_index": i,
                    "_total_chunks": num_chunks,
                    "_content_hash": doc["content_hash"],
                })

            progress.advance(task)

    console.print(
        f"[dim]Chunked {len(docs_to_ingest)} document(s) into "
        f"{len(all_texts)} chunk(s).[/dim]"
    )

    # --- Phase 2: Contextual retrieval (LLM prefixes) ---
    if config.ENABLE_CONTEXTUAL_RETRIEVAL:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating context prefixes", total=len(all_texts))
            for idx in range(len(all_texts)):
                prefix = generate_chunk_context(
                    all_texts[idx], all_metadatas[idx]["_doc_text"]
                )
                all_texts[idx] = f"{prefix}\n\n{all_texts[idx]}"
                progress.advance(task)

    # --- Phase 3: Build final metadata ---
    final_metadatas: list[dict] = []
    for idx, raw in enumerate(all_metadatas):
        if config.ENABLE_METADATA_ENRICHMENT and metadata_lookup:
            meta = extract_metadata(
                source=raw["_doc_source"],
                chunk_index=raw["_chunk_index"],
                total_chunks=raw["_total_chunks"],
                text=all_texts[idx],
                full_text=raw["_doc_text"],
                metadata_lookup=metadata_lookup,
            )
        else:
            meta = {
                "source": raw["_doc_source"],
                "chunk_index": raw["_chunk_index"],
            }
        meta["content_hash"] = raw["_content_hash"]
        final_metadatas.append(meta)

    # --- Phase 4: Embed & store ---
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
                metadatas=final_metadatas[start:end],
            )
            progress.advance(task, advance=min(batch_size, len(all_texts) - start))

    # Summary
    table = Table(title="Ingestion Summary", show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Documents processed", str(len(docs_to_ingest)))
    table.add_row("Documents skipped (unchanged)", str(skipped))
    table.add_row("Chunks created", str(len(all_texts)))
    table.add_row("Total in collection", str(collection.count()))
    console.print(table)


if __name__ == "__main__":
    ingest_documents()
