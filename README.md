# RAG System

A fully local Retrieval-Augmented Generation system. No external APIs, no costs, fully private.

Ingests text and markdown documents, stores embeddings in ChromaDB, and answers questions using a local Ollama LLM. Every enhancement is controlled by a feature flag in `rag/config.py` — with all flags off the system behaves as a vanilla RAG pipeline.

## Requirements

- Python (conda `ml` environment)
- [Ollama](https://ollama.com/) installed and running

## Setup

```bash
# Activate environment
conda activate ml

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Quick Start

```bash
# 1. Download test data (KJV Bible)
python main.py download

# 2. (Optional) Generate chapter & book summaries via LLM
python main.py generate-metadata

# 3. Ingest documents into vector database (incremental)
python main.py ingest

# 4. List ingested collections
python main.py collections

# 5. Ask a question
python main.py query "What happened on the first day of creation?"

# 6. Or start an interactive chat
python main.py chat

# 7. Run retrieval evaluation
python main.py eval
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py download` | Download KJV Bible scriptures as test data |
| `python main.py generate-metadata` | Generate chapter & book summaries via LLM |
| `python main.py generate-metadata --force` | Regenerate all summaries (ignores cache) |
| `python main.py ingest` | Ingest documents from `data/` (incremental — skips unchanged files) |
| `python main.py ingest --fresh` | Wipe collection and re-ingest everything |
| `python main.py collections` | List all ingested collections and show which is active |
| `python main.py query "question"` | Ask a one-shot question |
| `python main.py chat` | Interactive chat mode (type `quit` to exit) |
| `python main.py eval` | Run retrieval evaluation against the test set |

## Feature Flags

All flags live in `rag/config.py` and default to `False`. Enable them individually to layer improvements on top of the vanilla pipeline.

Changing any ingestion-time setting automatically creates a **separate collection** (the collection name is derived from a hash of the active settings). This means you can experiment with different configurations without losing previous results — run `python main.py collections` to see all versions side by side.

### Ingestion & Retrieval

| Flag | Description |
|------|-------------|
| `ENABLE_CONTEXT_AWARE_CHUNKING` | Paragraph-aware or section-aware chunking (set `CHUNKING_STRATEGY` to `"paragraph"` or `"section"`) |
| `ENABLE_SEMANTIC_CHUNKING` | Embedding-based chunking that splits on topic shifts (takes precedence over context-aware chunking) |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | LLM generates a contextual prefix per chunk during ingestion |
| `ENABLE_METADATA_ENRICHMENT` | Attaches volume, book, chapter, summaries, and structural metadata to each chunk |
| `ENABLE_HYBRID_SEARCH` | Combines dense vector search with BM25 keyword search via Reciprocal Rank Fusion |
| `ENABLE_RERANKING` | Cross-encoder reranking of candidates (fetches `RERANK_INITIAL_K` then reranks to `TOP_K`) |

### Query & Generation

| Flag | Description |
|------|-------------|
| `ENABLE_QUERY_REFORMULATION` | LLM rewrites the user query into an optimized search query before retrieval |
| `ENABLE_ENHANCED_PROMPT` | Richer system prompt with step-by-step reasoning, cross-document synthesis, and confidence signals |
| `ENABLE_CONTEXT_PRESENTATION` | Deduplicates near-identical chunks and reorders by best available score |

### Advanced Generation

| Flag | Description |
|------|-------------|
| `ENABLE_PROMPT_CHAINING` | Multi-step pipeline: extract key facts from chunks, then generate a final answer from those facts |

### Evaluation

| Flag | Description |
|------|-------------|
| `ENABLE_EVALUATION` | Enables the evaluation subsystem (`eval/test_set.json`) |

### Feature-Specific Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNKING_STRATEGY` | `"paragraph"` | `"sentence"`, `"paragraph"`, or `"section"` (requires `ENABLE_CONTEXT_AWARE_CHUNKING`) |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.5` | Cosine similarity threshold for semantic chunk splits (lower = bigger chunks) |
| `BM25_WEIGHT` | `1.0` | BM25 weight in RRF fusion |
| `RRF_K` | `60` | RRF smoothing constant |
| `RERANK_INITIAL_K` | `20` | Number of candidates to fetch before cross-encoder reranking |
| `CROSS_ENCODER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model for reranking |

## Using Your Own Documents

Place `.txt` or `.md` files in the `data/` directory, then run:

```bash
python main.py ingest
python main.py chat
```

Ingestion is incremental by default — only new or modified files are processed. Use `--fresh` to wipe and rebuild from scratch.

## Project Structure

```
project/
├── main.py              # Entry point
├── scripts/
│   ├── cli.py           # CLI commands
│   ├── download_scriptures.py
│   └── generate_metadata.py  # Two-pass LLM summarization
├── rag/
│   ├── config.py        # Configuration + feature flags
│   ├── chunking.py      # Chunking strategies (sentence, paragraph, section, semantic)
│   ├── ingest.py        # Document loading, chunking, enrichment, embedding
│   ├── retriever.py     # Vector search, BM25, hybrid search
│   ├── reranker.py      # RRF fusion + cross-encoder reranking
│   ├── query.py         # Query reformulation
│   ├── generator.py     # Prompt construction, enhanced prompts, prompt chaining
│   └── evaluation.py    # Retrieval metrics (Recall@k, Precision@k, MRR)
├── data/                # Your documents go here
│   └── metadata/        # Generated metadata (volume, book, chapter summaries)
├── eval/
│   └── test_set.json    # Q&A test pairs for evaluation
├── docs/
│   ├── plan_v2.md       # Feature flag implementation plan
│   ├── improvements.md  # RAG improvement techniques reference
│   └── RAG.md           # RAG techniques deep-dive
└── requirements.txt
```

## Documentation

- [plan_v2.md](docs/plan_v2.md) — Feature flags and phased improvement plan
- [improvements.md](docs/improvements.md) — Implementation guide for RAG techniques
- [RAG.md](docs/RAG.md) — Detailed reference on RAG improvement techniques
