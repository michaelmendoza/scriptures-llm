# Plan: LLM-Generated Book-Level Metadata (Two-Pass Summarization)

## Context

Volume-level `metadata.json` and `data/manifest.json` already exist (written by `download_scriptures.py`). This plan adds **per-book metadata** with LLM-generated summaries. Since this requires many ollama calls and takes significant time, it will be a **separate CLI command** (`python main.py generate-metadata`) rather than part of `download`.

**Two-pass approach**: Instead of sampling snippets from each book, we first summarize every chapter individually (1,189 calls), then synthesize each book summary from its chapter summaries (66 calls). This ensures no content is skipped and produces more accurate summaries.

All metadata lives under **`data/metadata/`** — a dedicated subfolder within `data/`. Safe because the ingest pipeline only globs `*.md` and `*.txt`, so JSON files are ignored.

---

## Data Layout

```
data/
  metadata/
    manifest.json                              # top-level manifest
    old-testament/
      metadata.json                            # volume metadata
      genesis.meta.json                        # book metadata (LLM-generated)
      chapters/
        genesis-1.summary.json                 # chapter summaries (LLM-generated)
        genesis-2.summary.json
        ...
    new-testament/
      metadata.json
      matthew.meta.json
      chapters/
        matthew-1.summary.json
        ...
  old-testament/
    genesis-1.md                               # raw chapter files (unchanged)
    ...
  new-testament/
    matthew-1.md
    ...
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `scripts/download_scriptures.py` | Move volume `metadata.json` and `manifest.json` output to `data/metadata/` |
| `scripts/generate_metadata.py` | **New file** — two-pass LLM summarization |
| `scripts/cli.py` | Add `cmd_generate_metadata()` and wire the `generate-metadata` command |
| `docs/plan_v2.md` | Add note about book-level metadata under Metadata Enrichment |

---

## Step 0: Move existing metadata output to `data/metadata/`

Update `scripts/download_scriptures.py`:
- Add `METADATA_DIR = DATA_DIR / "metadata"` constant
- `write_volume_metadata()` writes to `data/metadata/{volume-slug}/metadata.json`
- `write_manifest()` writes to `data/metadata/manifest.json`

---

## Step 1: New file `scripts/generate_metadata.py`

### Pass 1: Chapter summaries

**`_summarize_chapter(book_title, chapter_num, chapter_text)`**
- Calls `ollama.chat()` (non-streaming) with `config.LLM_MODEL`
- Prompt: "Summarize this chapter in 2-3 sentences, focusing on key events, themes, and people."
- Returns summary string

**`generate_chapter_summaries(force=False)`**
- For each volume → book → chapter:
  - Read chapter markdown from `data/{volume-slug}/{book-slug}-{n}.md`
  - Check if `data/metadata/{volume-slug}/chapters/{book-slug}-{n}.summary.json` exists — skip if not `force`
  - Call `_summarize_chapter()`
  - Write summary JSON
- Rich progress bar (1,189 chapters total)

**Chapter summary schema**: `data/metadata/{volume-slug}/chapters/{book-slug}-{n}.summary.json`
```json
{
  "book": "Genesis",
  "chapter": 1,
  "summary": "LLM-generated 2-3 sentence chapter summary...",
  "generated_with": "llama3.2",
  "generated_at": "2026-02-24T05:35:53Z"
}
```

### Pass 2: Book summaries from chapter summaries

**`_summarize_book(book_title, volume_title, chapter_summaries)`**
- Takes the list of chapter summary strings
- Calls `ollama.chat()` with all chapter summaries concatenated as context
- Prompt: "Given these chapter summaries, provide a concise 2-3 sentence summary of the entire book's themes, narrative arc, and significance."
- Returns summary string

**`generate_book_metadata(force=False)`**
- For each volume → book:
  - Collect all chapter summaries from `data/metadata/{volume-slug}/chapters/{book-slug}-*.summary.json`
  - Check if `data/metadata/{volume-slug}/{book-slug}.meta.json` exists — skip if not `force`
  - Call `_summarize_book()` with the collected chapter summaries
  - Write book metadata JSON
- Rich progress bar (66 books)

**Book metadata schema**: `data/metadata/{volume-slug}/{book-slug}.meta.json`
```json
{
  "schema_version": "1.0",
  "title": "Genesis",
  "slug": "genesis",
  "volume": "Old Testament",
  "chapters": 50,
  "verses": 1533,
  "summary": "LLM-generated 2-3 sentence book summary...",
  "chapter_summaries": [
    {"chapter": 1, "summary": "..."},
    {"chapter": 2, "summary": "..."}
  ],
  "generated_with": "llama3.2",
  "generated_at": "2026-02-24T05:35:53Z"
}
```

### `main(force=False)` entry point
- Reads volume metadata from `data/metadata/*/metadata.json` (must exist — run `download` first)
- Runs pass 1: `generate_chapter_summaries(force)`
- Runs pass 2: `generate_book_metadata(force)`
- Prints summary of how many chapters/books were generated vs skipped

---

## Step 2: Changes to `scripts/cli.py`

Add `cmd_generate_metadata()` function and wire `generate-metadata` command in `run_cli()`:
- Optional `--force` flag to regenerate all summaries
- Add to usage text

---

## Step 3: Update `docs/plan_v2.md`

Add a note under the Metadata Enrichment section about the book-level metadata files in `data/metadata/` and how the ingest pipeline can read them for chunk enrichment.

---

## Design Decisions

- **Two-pass summarization** — chapter summaries first, then book summaries from those. No content is skipped. Chapter summaries are individually small enough to fit full text in the LLM context. Book summaries are synthesized from comprehensive chapter-level understanding.
- **Per-chapter caching** — each chapter summary is saved individually. If the process is interrupted after 500/1189 chapters, re-running picks up where it left off. `--force` regenerates everything.
- **Chapter summaries embedded in book metadata** — the `chapter_summaries` array in `{book}.meta.json` preserves the intermediate work and is useful for ingest enrichment (each chapter chunk can reference its own summary).
- **All metadata in `data/metadata/`** — dedicated subfolder keeps all metadata (volume, book, chapter) organized separately from raw chapter files. The ingest pipeline only globs `*.md`/`*.txt`, so JSON files are automatically ignored.
- **~1,255 LLM calls total** — 1,189 chapters + 66 books. With caching, this only runs once unless `--force` is used.

---

## Verification

1. Run `python main.py download` — volume metadata now at `data/metadata/old-testament/metadata.json`, manifest at `data/metadata/manifest.json`
2. Run `python main.py generate-metadata` — should create 1,189 chapter summary files + 66 book meta files in `data/metadata/` with Rich progress
3. Check `data/metadata/old-testament/chapters/genesis-1.summary.json` — should have coherent chapter summary
4. Check `data/metadata/old-testament/genesis.meta.json` — should have book summary + `chapter_summaries` array
5. Run again without `--force` — should skip everything ("already exists")
6. Run with `--force` — should regenerate all
7. Run `python main.py ingest` — should still work unchanged (only reads `data/*.md`)
