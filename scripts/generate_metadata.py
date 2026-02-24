"""Generate LLM-powered chapter and book summaries via two-pass summarization."""

import json
from datetime import datetime, timezone
from pathlib import Path

import ollama
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from rag import config
from scripts.download_scriptures import METADATA_DIR, DATA_DIR, _make_slug

console = Console()


def _summarize_chapter(book_title: str, chapter_num: int, chapter_text: str) -> str:
    """Summarize a single chapter using the LLM."""
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biblical scholar. Summarize the given chapter "
                    "in 2-3 concise sentences, focusing on key events, themes, "
                    "and people. Do not include the chapter reference in your summary."
                ),
            },
            {
                "role": "user",
                "content": f"{book_title} Chapter {chapter_num}:\n\n{chapter_text}",
            },
        ],
    )
    return response["message"]["content"].strip()


def _summarize_book(book_title: str, volume_title: str, chapter_summaries: list[dict]) -> str:
    """Summarize an entire book from its chapter summaries."""
    summaries_text = "\n".join(
        f"Chapter {cs['chapter']}: {cs['summary']}" for cs in chapter_summaries
    )
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biblical scholar. Given the chapter-by-chapter summaries below, "
                    "provide a concise 2-3 sentence summary of the entire book's themes, "
                    "narrative arc, and significance. Do not list individual chapters."
                ),
            },
            {
                "role": "user",
                "content": f"{book_title} ({volume_title}):\n\n{summaries_text}",
            },
        ],
    )
    return response["message"]["content"].strip()


def generate_chapter_summaries(force: bool = False) -> int:
    """Pass 1: Summarize each chapter individually.

    Returns the number of chapters generated (vs skipped).
    """
    # Collect all chapters to process from volume metadata
    chapters_to_process = []
    for meta_path in sorted(METADATA_DIR.glob("*/metadata.json")):
        volume_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        volume_slug = volume_meta["slug"]
        volume_dir = DATA_DIR / volume_slug
        chapters_dir = METADATA_DIR / volume_slug / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        for book in volume_meta["books"]:
            for ch_num in range(1, book["chapters"] + 1):
                chapters_to_process.append({
                    "book_title": book["title"],
                    "book_slug": book["slug"],
                    "chapter_num": ch_num,
                    "volume_dir": volume_dir,
                    "chapters_dir": chapters_dir,
                })

    generated = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Summarizing chapters", total=len(chapters_to_process))

        for item in chapters_to_process:
            summary_path = item["chapters_dir"] / f"{item['book_slug']}-{item['chapter_num']}.summary.json"

            if summary_path.exists() and not force:
                skipped += 1
                progress.advance(task)
                continue

            # Read the chapter markdown
            md_path = item["volume_dir"] / f"{item['book_slug']}-{item['chapter_num']}.md"
            if not md_path.exists():
                progress.advance(task)
                continue

            chapter_text = md_path.read_text(encoding="utf-8")
            summary = _summarize_chapter(item["book_title"], item["chapter_num"], chapter_text)

            summary_data = {
                "book": item["book_title"],
                "chapter": item["chapter_num"],
                "summary": summary,
                "generated_with": config.LLM_MODEL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            summary_path.write_text(
                json.dumps(summary_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            generated += 1
            progress.advance(task)

    console.print(f"  Chapters: {generated} generated, {skipped} skipped (already exist)")
    return generated


def generate_book_metadata(force: bool = False) -> int:
    """Pass 2: Summarize each book from its chapter summaries.

    Returns the number of books generated (vs skipped).
    """
    books_to_process = []
    for meta_path in sorted(METADATA_DIR.glob("*/metadata.json")):
        volume_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        volume_slug = volume_meta["slug"]

        for book in volume_meta["books"]:
            books_to_process.append({
                "book_title": book["title"],
                "book_slug": book["slug"],
                "volume_title": volume_meta["volume_title"],
                "volume_slug": volume_slug,
                "chapters": book["chapters"],
                "verses": book["verses"],
            })

    generated = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Summarizing books", total=len(books_to_process))

        for item in books_to_process:
            book_meta_path = METADATA_DIR / item["volume_slug"] / f"{item['book_slug']}.meta.json"

            if book_meta_path.exists() and not force:
                skipped += 1
                progress.advance(task)
                continue

            # Collect chapter summaries
            chapters_dir = METADATA_DIR / item["volume_slug"] / "chapters"
            chapter_summaries = []
            for ch_num in range(1, item["chapters"] + 1):
                summary_path = chapters_dir / f"{item['book_slug']}-{ch_num}.summary.json"
                if summary_path.exists():
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    chapter_summaries.append({
                        "chapter": data["chapter"],
                        "summary": data["summary"],
                    })

            if not chapter_summaries:
                progress.advance(task)
                continue

            book_summary = _summarize_book(
                item["book_title"], item["volume_title"], chapter_summaries
            )

            book_data = {
                "schema_version": "1.0",
                "title": item["book_title"],
                "slug": item["book_slug"],
                "volume": item["volume_title"],
                "chapters": item["chapters"],
                "verses": item["verses"],
                "summary": book_summary,
                "chapter_summaries": chapter_summaries,
                "generated_with": config.LLM_MODEL,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            book_meta_path.write_text(
                json.dumps(book_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            generated += 1
            progress.advance(task)

    console.print(f"  Books: {generated} generated, {skipped} skipped (already exist)")
    return generated


def main(force: bool = False):
    """Run two-pass summarization: chapters first, then books."""
    # Verify metadata exists
    volume_metas = list(METADATA_DIR.glob("*/metadata.json"))
    if not volume_metas:
        console.print(
            "[red]No volume metadata found. Run 'python main.py download' first.[/red]"
        )
        return

    console.print(f"Found {len(volume_metas)} volume(s) in {METADATA_DIR}")

    console.print("\n[bold]Pass 1:[/bold] Chapter summaries")
    generate_chapter_summaries(force=force)

    console.print("\n[bold]Pass 2:[/bold] Book summaries")
    generate_book_metadata(force=force)


if __name__ == "__main__":
    main()
