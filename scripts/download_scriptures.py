"""Download KJV Bible JSON and convert to markdown files in data/."""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "https://raw.githubusercontent.com/beandog/lds-scriptures/master/json/kjv-scriptures-json.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
        volume_dir = DATA_DIR / volume.lower().replace(" ", "-")
        volume_dir.mkdir(parents=True, exist_ok=True)

        for book, chapters in books.items():
            for chapter_num, verses in sorted(chapters.items()):
                filename = f"{book.lower().replace(' ', '-')}-{chapter_num}.md"
                filepath = volume_dir / filename

                lines = [f"# {book} {chapter_num}\n"]
                for v in sorted(verses, key=lambda x: x["verse_number"]):
                    lines.append(f"**{v['verse_number']}** {v['scripture_text']}\n")

                filepath.write_text("\n".join(lines), encoding="utf-8")
                total_files += 1

        print(f"  {volume}: {sum(len(ch) for b in books.values() for ch in b.values())} verses across {sum(len(b) for b in books.values())} chapters")

    print(f"\nWrote {total_files} files to {DATA_DIR}/")


def main():
    data = download_json(URL)
    print(f"Downloaded {len(data)} verses")

    grouped = group_by_book_chapter(data)
    write_markdown_files(grouped)


if __name__ == "__main__":
    main()
