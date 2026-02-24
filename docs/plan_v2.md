# RAG System Feature Flags & Phased Improvements

## Context

The current RAG system is a "vanilla" implementation: sentence-boundary chunking, dense vector search only, single-shot generation. This plan adds 12 features across 4 phases, each controlled by a boolean feature flag in `config.py` (all defaulting to `False` for backward compatibility).

---

## Feature Flags (all in `config.py`)

```python
# Phase 1: Ingestion & Retrieval
ENABLE_CONTEXT_AWARE_CHUNKING = False
ENABLE_SEMANTIC_CHUNKING = False
ENABLE_CONTEXTUAL_RETRIEVAL = False
ENABLE_METADATA_ENRICHMENT = False
ENABLE_HYBRID_SEARCH = False
ENABLE_RERANKING = False

# Phase 2: Query & Generation
ENABLE_QUERY_REFORMULATION = False
ENABLE_ENHANCED_PROMPT = False
ENABLE_CONTEXT_PRESENTATION = False

# Phase 3: Advanced Generation
ENABLE_PROMPT_CHAINING = False

# Phase 4: Evaluation
ENABLE_EVALUATION = False
```

Plus feature-specific settings: `CHUNKING_STRATEGY`, `SEMANTIC_SIMILARITY_THRESHOLD`, `BM25_WEIGHT`, `RRF_K`, `RERANK_INITIAL_K`, `CROSS_ENCODER_MODEL`, `EVAL_DIR`.

---

## New Files to Create

| File | Purpose |
|------|---------|
| `rag/chunking.py` | Paragraph-aware, section-aware & semantic chunking strategies |
| `rag/reranker.py` | RRF fusion + cross-encoder reranking |
| `rag/query.py` | Query reformulation via LLM |
| `rag/evaluation.py` | Test set runner + retrieval metrics (Recall@k, Precision@k, MRR) |
| `eval/test_set.json` | Q&A test pairs from the KJV Bible data |

## Files to Modify

| File | Changes |
|------|---------|
| `rag/config.py` | Add all feature flags + feature-specific settings |
| `rag/ingest.py` | Use `chunking.py` dispatch, add `generate_chunk_context()`, add `extract_metadata()` |
| `rag/retriever.py` | Extract `_vector_search()`, add `bm25_search()`, modify `retrieve()` for hybrid/reranking |
| `rag/generator.py` | Add `ENHANCED_SYSTEM_PROMPT`, `_prepare_contexts()`, `analyze_and_extract()`, `generate_from_facts()` |
| `scripts/cli.py` | Wire query reformulation, prompt chaining, add `cmd_eval()`, update usage text |
| `requirements.txt` | Add `rank-bm25`, `sentence-transformers` |

---

## Phase 1: Ingestion & Retrieval

### 1. Context-Aware Chunking (`rag/chunking.py`)

Create new module with three strategies:
- **`chunk_sentences()`** — current logic, extracted from `ingest.py`
- **`chunk_paragraphs()`** — split on `\n\n`, accumulate paragraphs up to `CHUNK_SIZE`, overlap by including last paragraph of previous chunk. If a single paragraph exceeds size, fall back to sentence splitting.
- **`chunk_sections()`** — split on markdown headings (`^#{1,6}\s+`), each section = heading + content. If section > `CHUNK_SIZE`, sub-chunk with paragraph strategy. Always prepend section heading to every sub-chunk.
- **`chunk_text()`** — dispatcher that reads `config.CHUNKING_STRATEGY` ("sentence"|"paragraph"|"section")

In `ingest.py`: when `ENABLE_CONTEXT_AWARE_CHUNKING`, call `chunking.chunk_text()` with `config.CHUNKING_STRATEGY`; otherwise use original `chunk_text()`.

### 2. Semantic Chunking (`rag/chunking.py`)

Controlled by separate flag `ENABLE_SEMANTIC_CHUNKING` (takes precedence over `ENABLE_CONTEXT_AWARE_CHUNKING` when both are on).

- **`chunk_semantic()`** — uses the embedding model to detect topic shifts between consecutive sentences:
  1. Split text into sentences
  2. Embed each sentence using `OllamaEmbeddingFunction`
  3. Compute cosine similarity between consecutive sentence embeddings
  4. Identify split points where similarity drops below `SEMANTIC_SIMILARITY_THRESHOLD` (default 0.5)
  5. Group sentences between split points into chunks
  6. If a group exceeds `CHUNK_SIZE`, sub-split using sentence strategy
  7. If a group is too small, merge with the adjacent group that has higher similarity
- Config setting: `SEMANTIC_SIMILARITY_THRESHOLD = 0.5` — lower = fewer splits (bigger chunks), higher = more splits (smaller chunks)
- Trade-off: significantly increases ingestion time due to embedding every sentence individually. Add Rich progress indicator.
- In the dispatcher: when `ENABLE_SEMANTIC_CHUNKING`, use `chunk_semantic()` regardless of `CHUNKING_STRATEGY`

### 3. Metadata Enrichment (`rag/ingest.py`)

All metadata now lives under `data/metadata/` (see `scripts/download_scriptures.py` and `scripts/generate_metadata.py`):
- **Volume metadata**: `data/metadata/{volume-slug}/metadata.json` — volume descriptions, per-book stats, source attribution
- **Book metadata**: `data/metadata/{volume-slug}/{book-slug}.meta.json` — LLM-generated book summaries + chapter summaries (created by `python main.py generate-metadata`)
- **Chapter summaries**: `data/metadata/{volume-slug}/chapters/{book-slug}-{n}.summary.json` — per-chapter LLM summaries
- **Manifest**: `data/metadata/manifest.json` — top-level dataset summary

New `_load_metadata()` helper:
- In `load_documents()`, read `data/metadata/*/metadata.json` and `data/metadata/*/*.meta.json`
- Build a lookup: `{book_slug: {volume, book_title, total_chapters, book_summary, chapter_summaries, ...}}`
- Pass this lookup into `ingest_documents()` so it's available during chunk metadata assembly

New `extract_metadata()` function that enriches chunk metadata when `ENABLE_METADATA_ENRICHMENT`:
- **From volume metadata**: `volume` ("Old Testament"), `book` ("Genesis"), `book_slug` ("genesis"), `chapter` (parsed from filename)
- **From book metadata** (if available): `book_summary`, `chapter_summary` for the specific chapter
- **Source metadata**: `filename`, `directory`, `document_title` (from first `#` heading)
- **Structural**: `section_heading` (nearest heading above chunk), `total_chunks`, `position` (start/middle/end)

New helper `_find_section_heading()` to locate nearest heading above a chunk's position in the source document.

This enables **filtered retrieval** at query time — e.g., `collection.query(..., where={"volume": "New Testament"})` or `where={"book": "Genesis"}` to scope searches to specific parts of the Bible. Chapter summaries can also serve as contextual prefixes (see Contextual Retrieval below), reducing LLM calls at ingest time.

### 4. Contextual Retrieval (`rag/ingest.py`)

New `generate_chunk_context()` function:
- Sends chunk + first 3000 chars of parent document to LLM
- LLM returns 2-3 sentence contextual prefix
- Prefix is prepended to chunk text before embedding/storing
- When `ENABLE_CONTEXTUAL_RETRIEVAL` is on in `ingest_documents()`, each chunk gets this prefix
- Add Rich progress indicator since this is slow (one LLM call per chunk)

### 5. Hybrid Search (`rag/retriever.py`)

- Add `_get_bm25_index()` — lazily loads all docs from ChromaDB, tokenizes, builds `BM25Okapi` index, caches in module-level variable
- Add `bm25_search(query, top_k)` — tokenize query, score with BM25, return top-k with metadata
- Modify `retrieve()`: when `ENABLE_HYBRID_SEARCH`, run both `_vector_search()` and `bm25_search()`, merge with RRF from `reranker.py`
- Extract current search logic into `_vector_search()` helper

### 6. Reranking (`rag/reranker.py`)

- **`reciprocal_rank_fusion(*ranked_lists, k=60)`** — merge multiple ranked lists by RRF score, deduplicate by chunk ID
- **`cross_encoder_rerank(query, candidates, top_k)`** — lazily load `CrossEncoder` model, score all (query, chunk) pairs, sort by score, return top-k
- In `retrieve()`: when `ENABLE_RERANKING`, fetch `RERANK_INITIAL_K` (default 20) candidates, then rerank down to `TOP_K`
- When both hybrid + reranking are on: vector top-20 + BM25 top-20 → RRF merge → cross-encoder rerank → top-5

---

## Phase 2: Query & Generation

### 7. Query Reformulation (`rag/query.py`)

- `reformulate_query(user_query)` — LLM rewrites conversational query into optimized search query
- Wired in `scripts/cli.py` (not `retriever.py`) so original query goes to generator but reformulated query goes to retriever
- Display reformulated query to user as dim text

### 8. Enhanced Prompt (`rag/generator.py`)

- Add `ENHANCED_SYSTEM_PROMPT` constant with: step-by-step reasoning, cross-document synthesis, confidence signals, conflict acknowledgment
- In `build_prompt()`: select `ENHANCED_SYSTEM_PROMPT` when `ENABLE_ENHANCED_PROMPT`, otherwise use original `SYSTEM_PROMPT`

### 9. Context Presentation (`rag/generator.py`)

- Add `_prepare_contexts()` function:
  - **Deduplicate**: remove chunks with >90% Jaccard word similarity
  - **Order**: sort by best available score (`rerank_score` > `rrf_score` > `score`)
- When `ENABLE_METADATA_ENRICHMENT` is also on, format labels with document title + section heading
- Called in `build_prompt()` when `ENABLE_CONTEXT_PRESENTATION`

---

## Phase 3: Advanced Generation

### 10. Prompt Chaining (`rag/generator.py` + `scripts/cli.py`)

New functions in `generator.py`:
- **`analyze_and_extract(query, contexts)`** — LLM extracts key facts from retrieved chunks with source attributions
- **`generate_from_facts(query, facts)`** — LLM generates final answer from extracted facts (streaming)

In `cli.py`: when `ENABLE_PROMPT_CHAINING`, the pipeline becomes:
1. (Optional) Reformulate query
2. Retrieve chunks
3. Analyze/extract key facts
4. Generate from facts (streamed)

---

## Phase 4: Evaluation

### 11-12. Test Set & Retrieval Metrics (`rag/evaluation.py` + `eval/test_set.json`)

Create `eval/test_set.json` with 20+ Q&A pairs from KJV Bible data (mix of easy, hard, multi-hop).

`rag/evaluation.py`:
- `load_test_set()` — load JSON test pairs
- `recall_at_k()`, `precision_at_k()`, `mrr()` — metric functions
- `run_retrieval_eval()` — run all test questions through retrieval, compute aggregate metrics

Add `cmd_eval()` to CLI and `eval` command to `run_cli()`.

---

## New Dependencies

```
rank-bm25              # BM25 sparse search (Phase 1)
sentence-transformers   # Cross-encoder reranking (Phase 1)
```

Note: `sentence-transformers` pulls in PyTorch (~2GB). The cross-encoder model itself is small (~22MB, downloaded on first use).

---

## Implementation Order

Within each phase, features build on each other:

**Phase 1**: config flags → chunking.py (context-aware + semantic) → metadata enrichment → contextual retrieval → hybrid search (+ retriever refactor) → reranker.py

**Phase 2**: query.py → enhanced prompt → context presentation

**Phase 3**: prompt chaining functions → CLI wiring

**Phase 4**: evaluation.py → test_set.json → CLI eval command

---

## Verification

After each phase:
1. **Flags off**: Run `python main.py ingest` then `python main.py query "Who is Jesus?"` — should behave identically to current system
2. **Flags on**: Enable the phase's flags in `config.py`, re-ingest, re-query — verify new behavior
3. **Phase 4**: Run `python main.py eval` and check metric output
