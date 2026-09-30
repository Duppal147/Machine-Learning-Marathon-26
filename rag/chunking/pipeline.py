"""Compose loader -> block filters -> chunker -> chunk filters -> enrichers.

Named configurations live in REGISTRY so experiments are selectable by name
(`build_pipeline("hybrid", max_tokens=300)`); add a factory to try a new combination.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .chunkers import Chunker, FixedSizeChunker, HybridChunker, SectionChunker
from .enrichers import CorpusMetadataEnricher, Enricher, HeadingPrefixEnricher
from .filters import BlockFilter, ChunkFilter, DropSections, MinTokens
from .loaders import DoclingJsonLoader, DocumentLoader
from .models import Chunk, Document


class AutoLoader:
    """Dispatch to a loader by file suffix, e.g. add ".md": MarkdownLoader() later."""

    def __init__(self, loaders: dict[str, DocumentLoader] | None = None):
        self.loaders = loaders or {".json": DoclingJsonLoader()}

    def load(self, path: Path) -> Document:
        path = Path(path)
        if path.suffix not in self.loaders:
            raise ValueError(f"no loader registered for {path.suffix!r} ({path})")
        return self.loaders[path.suffix].load(path)


@dataclass
class ChunkingPipeline:
    name: str
    chunker: Chunker
    loader: DocumentLoader = field(default_factory=AutoLoader)
    enrichers: list[Enricher] = field(default_factory=list)
    block_filters: list[BlockFilter] = field(default_factory=list)
    chunk_filters: list[ChunkFilter] = field(default_factory=list)

    def run_document(self, doc: Document) -> list[Chunk]:
        blocks = [b for b in doc.blocks if all(f.keep(b) for f in self.block_filters)]
        chunks = [c for c in self.chunker.chunk(blocks) if all(f.keep(c) for f in self.chunk_filters)]

        out = []
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{doc.doc_id}::{self.name}::{i:04d}"
            chunk.strategy = self.name
            chunk.metadata = {"title": doc.title, **doc.metadata, **chunk.metadata}
            for enricher in self.enrichers:
                chunk = enricher.enrich(chunk, doc)
            out.append(chunk)
        return out

    def run(self, paths: Iterable[Path]) -> list[Chunk]:
        return [c for path in paths for c in self.run_document(self.loader.load(path))]


def _defaults(heading_prefix: bool) -> dict:
    enrichers = [CorpusMetadataEnricher()] + ([HeadingPrefixEnricher()] if heading_prefix else [])
    return {"enrichers": enrichers, "block_filters": [DropSections()], "chunk_filters": [MinTokens()]}


REGISTRY: dict[str, Callable[..., ChunkingPipeline]] = {
    # section-aware packing + heading path in the embedding (recommended default)
    "hybrid": lambda max_tokens=400: ChunkingPipeline(
        "hybrid", HybridChunker(max_tokens=max_tokens), **_defaults(heading_prefix=True)),
    # same chunks, no heading prefix: A/B the enrichment on its own
    "hybrid_raw": lambda max_tokens=400: ChunkingPipeline(
        "hybrid_raw", HybridChunker(max_tokens=max_tokens), **_defaults(heading_prefix=False)),
    "section": lambda max_tokens=1200: ChunkingPipeline(
        "section", SectionChunker(max_tokens=max_tokens), **_defaults(heading_prefix=True)),
    "fixed": lambda max_tokens=300: ChunkingPipeline(
        "fixed", FixedSizeChunker(max_tokens=max_tokens), **_defaults(heading_prefix=False)),
}


def build_pipeline(name: str, **kwargs) -> ChunkingPipeline:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**{k: v for k, v in kwargs.items() if v is not None})
