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
ENABLE_SEMANTIC_CHUNKING = True
ENABLE_CONTEXTUAL_RETRIEVAL = True
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

def _ingestion_settings_hash() -> str:
    """Short hash of all settings that affect ingested data."""
    parts = [
        f"embed={EMBEDDING_MODEL}",
        f"chunk_size={CHUNK_SIZE}",
        f"overlap={CHUNK_OVERLAP}",
        f"semantic={ENABLE_SEMANTIC_CHUNKING}",
        f"ctx_aware={ENABLE_CONTEXT_AWARE_CHUNKING}",
        f"strategy={CHUNKING_STRATEGY}",
        f"sem_thresh={SEMANTIC_SIMILARITY_THRESHOLD}",
        f"ctx_retr={ENABLE_CONTEXTUAL_RETRIEVAL}",
        f"llm={LLM_MODEL}",
        f"meta={ENABLE_METADATA_ENRICHMENT}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]
    return digest


def get_collection_name() -> str:
    """Return a deterministic collection name based on current settings."""
    return f"{COLLECTION_BASE_NAME}_{_ingestion_settings_hash()}"


COLLECTION_NAME = get_collection_name()
