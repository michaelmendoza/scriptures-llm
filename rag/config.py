from pathlib import Path

# Resolve paths relative to project root (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = _PROJECT_ROOT / "data"
CHROMA_DIR = _PROJECT_ROOT / "chroma_db"

# ChromaDB
COLLECTION_NAME = "documents"

# Chunking
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# Retrieval
TOP_K = 5
