# Vanilla RAG System Plan

## Overview
A fully local Retrieval-Augmented Generation system. Ingests text and markdown documents, stores embeddings in ChromaDB, and answers questions using a local Ollama LLM. No external APIs, no costs, fully private.

## Architecture

### Ingestion Pipeline
```
 Source Documents
 (data/*.txt, *.md)
        │
        ▼
 ┌──────────────┐
 │   Chunker    │
 │  (ingest.py) │
 └──────┬───────┘
        │  overlapping chunks
        ▼
 ┌──────────────┐
 │  Embeddings  │
 │  (Ollama)    │
 └──────┬───────┘
        │  vectors + metadata
        ▼
 ┌──────────────┐
 │   ChromaDB   │
 │  (chroma_db/)│
 └──────────────┘
```

### Query Pipeline
```
 User Question
        │
        ▼
 ┌──────────────┐     ┌──────────────────┐
 │  Retriever   │────►│    ChromaDB      │
 │  (search)    │◄────│  (vector store)  │
 └──────┬───────┘     └──────────────────┘
        │
        │  top-k chunks
        ▼
 ┌──────────────┐
 │  Generator   │
 │  (Ollama)    │
 └──────┬───────┘
        │
        ▼
    Responses 
```

## Project Structure
```
project/
├── data/                # Place your .txt and .md files here
├── chroma_db/           # ChromaDB persistent storage (auto-created)
├── docs/
│   ├── plan.md          # This file
│   └── RAG.md           # Detailed RAG improvement techniques
├── rag/
│   ├── config.py        # Configuration constants
│   ├── ingest.py        # Document loading, chunking, embedding
│   ├── retriever.py     # Semantic search against ChromaDB
│   └── generator.py     # Prompt construction + Ollama generation
├── main.py              # CLI entry point
└── requirements.txt     # Python dependencies
```

## Setup
1. Activate environment: `conda activate ml`
2. Install dependencies: `pip install -r requirements.txt`
3. Pull Ollama models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
4. Place documents in `data/`

## Usage
```bash
# Ingest documents into the vector database
python main.py ingest

# Ask a one-shot question
python main.py query "What is the refund policy?"

# Start interactive chat
python main.py chat
```

## Configuration (config.py)
| Setting | Default | Description |
|---------|---------|-------------|
| DATA_DIR | ./data | Directory containing source documents |
| CHROMA_DIR | ./chroma_db | ChromaDB persistent storage |
| CHUNK_SIZE | 1000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| EMBEDDING_MODEL | nomic-embed-text | Ollama embedding model |
| LLM_MODEL | llama3.2 | Ollama chat model |
| TOP_K | 5 | Number of chunks to retrieve |

## How It Works

### Ingestion
1. Scans `data/` recursively for `.txt` and `.md` files
2. Splits each file into overlapping chunks (1000 chars, 200 overlap) at sentence boundaries
3. Embeds each chunk using Ollama's `nomic-embed-text` model
4. Stores chunks + metadata (source filename, chunk index) in ChromaDB

### Querying
1. User question is embedded using the same embedding model
2. ChromaDB performs similarity search, returning the top-k most relevant chunks
3. Retrieved chunks are formatted into a prompt with source attribution
4. Ollama generates a streamed response grounded in the retrieved context
5. Answer includes `[source: filename]` citations

## This is Vanilla RAG
This system implements the simplest RAG architecture:
```
Query -> Embed -> Vector Search -> Context Stuffing -> LLM -> Answer
```
It uses a single retrieval pass, fixed chunking, and direct context injection. This is intentional — it provides a solid, understandable foundation.

## Suggestions for Improving RAG

See [RAG.md](RAG.md) for detailed explanations of each technique.

### High Impact

| Technique | Summary |
|-----------|---------|
| Cross-Encoder Re-Ranker | Re-score top-N retrieved chunks with a cross-encoder for much more precise relevance ranking |
| Contextual Retrieval | Prepend LLM-generated context to each chunk before embedding — up to 49% fewer retrieval failures |
| Reciprocal Rank Fusion | Combine semantic + keyword (BM25) search results into one superior ranked list |
| Knowledge Graphs | Extract entities and relationships for structured multi-hop reasoning across documents |

### Medium Impact

| Technique | Summary |
|-----------|---------|
| Context-Aware Chunking | Split at sentence, paragraph, or section boundaries instead of fixed character counts |
| HyDE | Generate a hypothetical answer and embed that instead of the raw query for better retrieval |
| Parent-Child Chunking | Retrieve small chunks for precision, pass larger parent context to the LLM |
| Conversation Memory | Use chat history to reformulate ambiguous follow-up queries |

### Lower Impact / Situational

| Technique | Summary |
|-----------|---------|
| Query Expansion | Rephrase the query multiple ways to improve recall |
| Fine-tuned Embeddings | Train embeddings on your domain data for better terminology capture |
| Agentic RAG | Let the LLM dynamically decide when and what to retrieve |
| Multi-step Retrieval | Iteratively retrieve more context if the first pass isn't sufficient |

### Evaluation

| Technique | Summary |
|-----------|---------|
| Test Set | Build 20-50 Q&A pairs from your documents to measure quality over time |
| Retrieval Metrics | Track recall@k, precision@k, and MRR for retrieved chunks |
| LLM-as-Judge | Use a second LLM call to detect hallucination and unfaithful answers |
| RAGAS Framework | Automated evaluation of faithfulness, relevance, and context precision |
