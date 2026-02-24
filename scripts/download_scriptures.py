"""Download KJV Bible JSON and convert to markdown files in data/."""

import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

URL = "https://raw.githubusercontent.com/beandog/lds-scriptures/master/json/kjv-scriptures-json.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_DIR = DATA_DIR / "metadata"

VOLUME_DESCRIPTIONS = {
    "Old Testament": (
        "The Old Testament of the King James Version (KJV) Bible, "
        "containing 39 books from Genesis through Malachi. Covers creation, "
        "the law, history of Israel, wisdom literature, and prophetic writings."
    ),
    "New Testament": (
        "The New Testament of the King James Version (KJV) Bible, "
        "containing 27 books from Matthew through Revelation. Covers the life "
        "and teachings of Jesus Christ, the early church, epistles, and prophecy."
    ),
}


def _make_slug(name: str) -> str:
    """Convert a title like 'Old Testament' to 'old-testament'."""
    return name.lower().replace(" ", "-")


def download_json(url: str) -> list[dict]:
    print(f"Downloading from {url}...")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def group_by_book_chapter(verses: list[dict]) -> dict:
    """Group verses into {volume: {book: {chapter: [verses]}}}."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for v in verses:
        grouped[v["volume_title"]][v["book_title"]][v["chapter_number"]].append(v)
    return grouped


def write_markdown_files(grouped: dict):
    """Write one markdown file per book chapter."""
    total_files = 0

    for volume, books in grouped.items():
        volume_dir = DATA_DIR / _make_slug(volume)
        volume_dir.mkdir(parents=True, exist_ok=True)

        for book, chapters in books.items():
            for chapter_num, verses in sorted(chapters.items()):
                filename = f"{_make_slug(book)}-{chapter_num}.md"
                filepath = volume_dir / filename

                lines = [f"# {book} {chapter_num}\n"]
                for v in sorted(verses, key=lambda x: x["verse_number"]):
                    lines.append(f"**{v['verse_number']}** {v['scripture_text']}\n")

                filepath.write_text("\n".join(lines), encoding="utf-8")
                total_files += 1

        print(f"  {volume}: {sum(len(ch) for b in books.values() for ch in b.values())} verses across {sum(len(b) for b in books.values())} chapters")

    print(f"\nWrote {total_files} files to {DATA_DIR}/")


def write_volume_metadata(grouped: dict) -> list[dict]:
    """Write a metadata.json file in each volume directory.

    Returns a list of volume summary dicts for the top-level manifest.
    """
    now = datetime.now(timezone.utc).isoformat()
    volume_summaries = []

    for volume_title, books in grouped.items():
        volume_slug = _make_slug(volume_title)
        meta_volume_dir = METADATA_DIR / volume_slug
        meta_volume_dir.mkdir(parents=True, exist_ok=True)

        book_entries = []
        total_chapters = 0
        total_verses = 0

        for book_title, chapters in books.items():
            book_slug = _make_slug(book_title)
            chapter_numbers = sorted(chapters.keys())
            num_chapters = len(chapter_numbers)
            num_verses = sum(len(verses) for verses in chapters.values())

            book_entries.append({
                "title": book_title,
                "slug": book_slug,
                "chapters": num_chapters,
                "verses": num_verses,
                "first_file": f"{book_slug}-{chapter_numbers[0]}.md",
                "last_file": f"{book_slug}-{chapter_numbers[-1]}.md",
            })

            total_chapters += num_chapters
            total_verses += num_verses

        metadata = {
            "schema_version": "1.0",
            "volume_title": volume_title,
            "slug": volume_slug,
            "description": VOLUME_DESCRIPTIONS.get(
                volume_title,
                f"Volume: {volume_title} from the King James Version Bible.",
            ),
            "source": {
                "translation": "King James Version (KJV)",
                "url": URL,
                "downloaded_at": now,
            },
            "statistics": {
                "total_books": len(book_entries),
                "total_chapters": total_chapters,
                "total_verses": total_verses,
            },
            "books": book_entries,
        }

        metadata_path = meta_volume_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Wrote {metadata_path}")

        volume_summaries.append({
            "title": volume_title,
            "slug": volume_slug,
            "metadata_file": f"{volume_slug}/metadata.json",
            "total_books": len(book_entries),
            "total_chapters": total_chapters,
            "total_verses": total_verses,
        })

    return volume_summaries


def write_manifest(volume_summaries: list[dict]):
    """Write a top-level manifest.json summarizing all volumes."""
    now = datetime.now(timezone.utc).isoformat()

    manifest = {
        "schema_version": "1.0",
        "dataset": "KJV Bible",
        "description": "The complete King James Version Bible, organized by volume.",
        "source": {
            "translation": "King James Version (KJV)",
            "url": URL,
            "downloaded_at": now,
        },
        "generated_at": now,
        "statistics": {
            "total_volumes": len(volume_summaries),
            "total_books": sum(v["total_books"] for v in volume_summaries),
            "total_chapters": sum(v["total_chapters"] for v in volume_summaries),
            "total_verses": sum(v["total_verses"] for v in volume_summaries),
        },
        "volumes": volume_summaries,
    }

    manifest_path = METADATA_DIR / "manifest.json"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Wrote {manifest_path}")


def main():
    data = download_json(URL)
    print(f"Downloaded {len(data)} verses")

    grouped = group_by_book_chapter(data)
    write_markdown_files(grouped)

    print("\nWriting metadata...")
    volume_summaries = write_volume_metadata(grouped)
    write_manifest(volume_summaries)
    print("Metadata complete.")


if __name__ == "__main__":
    main()
