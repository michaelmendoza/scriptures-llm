# Vanilla RAG System

A fully local Retrieval-Augmented Generation system. No external APIs, no costs, fully private.

Ingests text and markdown documents, stores embeddings in ChromaDB, and answers questions using a local Ollama LLM.

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

# 3. Ingest documents into vector database
python main.py ingest

# 4. Ask a question
python main.py query "What happened on the first day of creation?"

# 5. Or start an interactive chat
python main.py chat
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py download` | Download KJV Bible scriptures as test data |
| `python main.py generate-metadata` | Generate chapter & book summaries via LLM |
| `python main.py generate-metadata --force` | Regenerate all summaries (ignores cache) |
| `python main.py ingest` | Ingest all documents from `data/` into ChromaDB |
| `python main.py query "question"` | Ask a one-shot question |
| `python main.py chat` | Interactive chat mode (type `quit` to exit) |

## Using Your Own Documents

Place `.txt` or `.md` files in the `data/` directory, then run:

```bash
python main.py ingest
python main.py chat
```

## Project Structure

```
project/
├── main.py              # Entry point
├── scripts/
│   ├── cli.py           # CLI commands
│   ├── download_scriptures.py
│   └── generate_metadata.py  # Two-pass LLM summarization
├── rag/
│   ├── config.py        # Configuration
│   ├── ingest.py        # Document chunking + embedding
│   ├── retriever.py     # Semantic search
│   └── generator.py     # Prompt construction + LLM
├── data/                # Your documents go here
│   └── metadata/        # Generated metadata (volume, book, chapter summaries)
├── docs/
│   ├── plan.md          # Architecture overview
│   └── RAG.md           # RAG improvement techniques
└── requirements.txt
```

## Documentation

- [plan.md](docs/plan.md) — Architecture, configuration, and improvement suggestions
- [RAG.md](docs/RAG.md) — Detailed reference on RAG improvement techniques
