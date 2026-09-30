"""Filters drop content that hurts retrieval.

Block filters run before chunking (so even section-blind chunkers skip e.g. the
bibliography); chunk filters run after.
"""
import re
from typing import Protocol

from .chunkers import LenFn, approx_tokens
from .models import Block, Chunk


class BlockFilter(Protocol):
    def keep(self, block: Block) -> bool: ...


class ChunkFilter(Protocol):
    def keep(self, chunk: Chunk) -> bool: ...


DEFAULT_DROPPED_SECTIONS = frozenset({
    "references", "bibliography", "citations and bibliography",
    "contents", "table of contents", "list of figures", "list of tables",
})

_NUMBERING = re.compile(r"^(?:[IVXLC]+|[A-Z]|\d+(?:\.\d+)*)\.?\s+")


def section_name(heading: str) -> str:
    return _NUMBERING.sub("", heading).strip(" :").lower()


class DropSections:
    """Drop blocks under any heading whose (unnumbered, lowercased) name is in `names`."""

    def __init__(self, names=DEFAULT_DROPPED_SECTIONS):
        self.names = frozenset(names)

    def keep(self, block: Block) -> bool:
        return not any(section_name(h) in self.names for h in block.heading_path)


class MinTokens:
    def __init__(self, min_tokens: int = 5, len_fn: LenFn = approx_tokens):
        self.min_tokens = min_tokens
        self.len_fn = len_fn

    def keep(self, chunk: Chunk) -> bool:
        return self.len_fn(chunk.text) >= self.min_tokens
