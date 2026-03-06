import hashlib
from pathlib import Path

# Resolve paths relative to project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = _PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
CHROMA_DIR = _PROJECT_ROOT / "chroma_db"
EVAL_DIR = _PROJECT_ROOT / "eval"

# ChromaDB
COLLECTION_BASE_NAME = "documents"

# Chunking
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Retrieval
TOP_K = 5

# ---------------------------------------------------------------------------
# Feature Flags (all default to False for backward compatibility)
# ---------------------------------------------------------------------------

# Ingestion & Retrieval
ENABLE_CONTEXT_AWARE_CHUNKING = True
ENABLE_SEMANTIC_CHUNKING = False
ENABLE_CONTEXTUAL_RETRIEVAL = False
ENABLE_METADATA_ENRICHMENT = True
ENABLE_HYBRID_SEARCH = True
ENABLE_RERANKING = True

# Query & Generation
ENABLE_QUERY_REFORMULATION = True
ENABLE_ENHANCED_PROMPT = True
ENABLE_CONTEXT_PRESENTATION = True

# Advanced Generation
ENABLE_PROMPT_CHAINING = True

# Evaluation
ENABLE_EVALUATION = True

# ---------------------------------------------------------------------------
# Feature-specific settings
# ---------------------------------------------------------------------------

# Context-aware chunking strategy: "sentence" | "paragraph" | "section"
CHUNKING_STRATEGY = "paragraph"

# Semantic chunking: similarity threshold (lower = fewer splits / bigger chunks)
SEMANTIC_SIMILARITY_THRESHOLD = 0.5

# Hybrid search: BM25 weight in RRF fusion
BM25_WEIGHT = 1.0
RRF_K = 60

# Reranking: fetch more candidates, then rerank down to TOP_K
RERANK_INITIAL_K = 20
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Derived collection name
# ---------------------------------------------------------------------------

def get_ingestion_settings() -> dict:
    """Return ingestion-relevant settings as a dict.

    Values are strings/ints/floats/bools so they can be stored directly
    as ChromaDB collection metadata.
    """
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "semantic_chunking": ENABLE_SEMANTIC_CHUNKING,
        "context_aware_chunking": ENABLE_CONTEXT_AWARE_CHUNKING,
        "chunking_strategy": CHUNKING_STRATEGY,
        "semantic_threshold": SEMANTIC_SIMILARITY_THRESHOLD,
        "contextual_retrieval": ENABLE_CONTEXTUAL_RETRIEVAL,
        "metadata_enrichment": ENABLE_METADATA_ENRICHMENT,
    }


def _ingestion_settings_hash() -> str:
    """Short hash of all settings that affect ingested data."""
    settings = get_ingestion_settings()
    key = "|".join(f"{k}={v}" for k, v in sorted(settings.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def get_collection_name() -> str:
    """Return a deterministic collection name based on current settings."""
    return f"{COLLECTION_BASE_NAME}_{_ingestion_settings_hash()}"


COLLECTION_NAME = get_collection_name()
