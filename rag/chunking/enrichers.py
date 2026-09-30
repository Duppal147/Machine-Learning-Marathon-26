"""Enrichers add to a chunk's `embed_text` (recall tricks) or `metadata`; `text` is never touched.

Ideas that fit here later: an LLMContextEnricher that asks the chat model for a
one-sentence "where this chunk sits in the document" blurb (contextual retrieval).
"""
import csv
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .models import Chunk, Document

WATTBOT_METADATA = Path("WattBot2026/metadata_downloaded.csv")


class Enricher(Protocol):
    def enrich(self, chunk: Chunk, doc: Document) -> Chunk: ...


class HeadingPrefixEnricher:
    """Prefix `Doc title > Section > Subsection` so the embedding carries the chunk's location."""

    def __init__(self, include_title: bool = True, sep: str = " > "):
        self.include_title = include_title
        self.sep = sep

    def enrich(self, chunk: Chunk, doc: Document) -> Chunk:
        path = ([doc.title] if self.include_title else []) + list(chunk.heading_path)
        if not path:
            return chunk
        return replace(chunk, embed_text=f"{self.sep.join(path)}\n\n{chunk.embed_text}")


class CorpusMetadataEnricher:
    """Attach WattBot corpus metadata (ref_id, url, year, type, title) by joining on the file stem.

    WattBot answers cite `ref_id` (e.g. "amazon2023"), while doc_id is the PDF file stem
    (e.g. "2023_Amazon_Sustainability_Report_0186f6c3"); `local_path` in the CSV links them.
    A missing CSV or unknown doc leaves the chunk unchanged.
    """

    FIELDS = {"id": "ref_id", "url": "url", "year": "year", "type": "doc_type", "title": "ref_title"}

    def __init__(self, csv_path: Path = WATTBOT_METADATA):
        self.by_stem: dict[str, dict] = {}
        if Path(csv_path).exists():
            with open(csv_path, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("local_path"):
                        self.by_stem[Path(row["local_path"]).stem] = row

    def enrich(self, chunk: Chunk, doc: Document) -> Chunk:
        row = self.by_stem.get(chunk.doc_id)
        if row is None:
            return chunk
        extra = {dst: row[src] for src, dst in self.FIELDS.items() if row.get(src)}
        return replace(chunk, metadata={**chunk.metadata, **extra})
