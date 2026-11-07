#!/usr/bin/env python3
"""Check that parsed arXiv papers are the versions metadata.csv pins.

Usage:
    python scripts/check_corpus_versions.py [--metadata WattBot2026/metadata.csv]

Reads the arXiv stamp printed in each parsed document. Exits 1 on any mismatch. Replace a mismatched
PDF, then re-run parse_pdfs.py, describe_figures.py, chunk_docs.py and vector_index.py for that paper.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

PARSED_DIR = Path("documents/parsed_json")
PINNED = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)(v\d+)")


def compare(url: str, parsed_text: str) -> str:
    """Return match, mismatch vN, no-stamp, or unpinned for one document."""
    pinned = PINNED.search(url)
    if not pinned:
        return "unpinned"
    stamp = re.search(re.escape(pinned.group(1)) + r"(v\d+)", parsed_text)
    if not stamp:
        return "no-stamp"
    return "match" if stamp.group(1) == pinned.group(2) else f"mismatch {stamp.group(1)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=Path("WattBot2026/metadata.csv"))
    args = parser.parse_args()
    counts, failed = {}, False
    with args.metadata.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            pinned = PINNED.search(row["url"])
            path = PARSED_DIR / f"{pinned.group(1)}.json" if pinned else None
            if pinned and not path.exists():
                result = "not-parsed"
            else:
                result = compare(row["url"], path.read_text(errors="ignore") if path else "")
            counts[result.split()[0]] = counts.get(result.split()[0], 0) + 1
            if result.startswith(("mismatch", "not-parsed")):
                failed = True
                print(f"{row['id']}: pinned {row['url']} -> {result}")
    print(counts)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
