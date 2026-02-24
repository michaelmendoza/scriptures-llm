"""Chunking strategies for the RAG ingestion pipeline.

Provides sentence, paragraph, section, and semantic chunking.
The ``chunk_text`` dispatcher selects the strategy based on feature flags.
"""

import re

import numpy as np
import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rag import config

console = Console()


# ---------------------------------------------------------------------------
# Sentence chunking (original strategy, extracted from ingest.py)
# ---------------------------------------------------------------------------

def chunk_sentences(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
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

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Paragraph chunking
# ---------------------------------------------------------------------------

def chunk_paragraphs(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* on blank lines, accumulate paragraphs up to *chunk_size*.

    Overlap is achieved by including the last paragraph of the previous chunk
    at the start of the next.  If a single paragraph exceeds *chunk_size*,
    fall back to sentence splitting for that paragraph.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks: list[str] = []
    current_chunk = ""
    last_paragraph = ""

    for para in paragraphs:
        # If a single paragraph is too large, sub-split with sentences
        if len(para) > chunk_size:
            # Flush current accumulator first
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            sub_chunks = chunk_sentences(para, chunk_size, overlap)
            chunks.extend(sub_chunks)
            last_paragraph = sub_chunks[-1] if sub_chunks else ""
            continue

        test = current_chunk + "\n\n" + para if current_chunk else para
        if len(test) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: start next chunk with the last paragraph
            current_chunk = last_paragraph + "\n\n" + para if last_paragraph else para
        else:
            current_chunk = test

        last_paragraph = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------------------------------------------
# Section chunking (markdown heading aware)
# ---------------------------------------------------------------------------

def chunk_sections(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* on markdown headings (``# …``).

    Each section is heading + content.  If a section exceeds *chunk_size*, it
    is sub-chunked with the paragraph strategy and the heading is prepended to
    every sub-chunk.
    """
    # Split while keeping the heading delimiter
    parts = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)

    sections: list[tuple[str, str]] = []  # (heading, body)
    current_heading = ""
    current_body = ""

    for part in parts:
        if re.match(r'^#{1,6}\s+', part):
            # Save previous section
            if current_heading or current_body.strip():
                sections.append((current_heading, current_body.strip()))
            current_heading = part.strip()
            current_body = ""
        else:
            current_body += part

    # Don't forget last section
    if current_heading or current_body.strip():
        sections.append((current_heading, current_body.strip()))

    chunks: list[str] = []
    for heading, body in sections:
        section_text = (heading + "\n\n" + body).strip() if heading else body
        if len(section_text) <= chunk_size:
            if section_text:
                chunks.append(section_text)
        else:
            # Sub-chunk with paragraph strategy, prepend heading
            sub_chunks = chunk_paragraphs(body, chunk_size - len(heading) - 2, overlap)
            for sc in sub_chunks:
                chunk = (heading + "\n\n" + sc).strip() if heading else sc
                chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking (embedding-based topic shift detection)
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the configured Ollama embedding model."""
    embeddings = []
    for t in texts:
        resp = ollama.embed(model=config.EMBEDDING_MODEL, input=t)
        embeddings.append(resp["embeddings"][0])
    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def chunk_semantic(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Split *text* by detecting topic shifts between consecutive sentences.

    1. Split text into sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive embeddings
    4. Split where similarity drops below *threshold*
    5. Merge groups that are too small with the more-similar neighbour
    6. Sub-split groups that exceed *chunk_size*
    """
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    console.print("[dim]  Embedding sentences for semantic chunking...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding sentences", total=len(sentences))
        embeddings = []
        valid_sentences = []
        for s in sentences:
            resp = ollama.embed(model=config.EMBEDDING_MODEL, input=s)
            if not resp.get("embeddings"):
                progress.advance(task)
                continue
            valid_sentences.append(s)
            embeddings.append(resp["embeddings"][0])
            progress.advance(task)
        sentences = valid_sentences

    # Compute consecutive similarities
    similarities = [
        _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Identify split points
    split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # Build groups
    groups: list[list[str]] = []
    prev = 0
    for idx in split_indices:
        groups.append(sentences[prev:idx])
        prev = idx
    groups.append(sentences[prev:])

    # Merge very small groups (< 20% of chunk_size) with more similar neighbour
    min_group_size = chunk_size // 5
    merged: list[list[str]] = []
    for group in groups:
        group_text = " ".join(group)
        if merged and len(group_text) < min_group_size:
            merged[-1].extend(group)
        else:
            merged.append(group)

    # Convert groups to chunks, sub-splitting if too large
    chunks: list[str] = []
    for group in merged:
        group_text = " ".join(group).strip()
        if not group_text:
            continue
        if len(group_text) <= chunk_size:
            chunks.append(group_text)
        else:
            chunks.extend(chunk_sentences(group_text, chunk_size))

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Select chunking strategy based on feature flags and config.

    Priority: semantic > context-aware (paragraph/section) > sentence (default).
    """
    if config.ENABLE_SEMANTIC_CHUNKING:
        return chunk_semantic(text, chunk_size)

    if config.ENABLE_CONTEXT_AWARE_CHUNKING:
        strategy = config.CHUNKING_STRATEGY.lower()
        if strategy == "paragraph":
            return chunk_paragraphs(text, chunk_size, overlap)
        elif strategy == "section":
            return chunk_sections(text, chunk_size, overlap)
        else:
            return chunk_sentences(text, chunk_size, overlap)

    # Default: original sentence chunking
    return chunk_sentences(text, chunk_size, overlap)
