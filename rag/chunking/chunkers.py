"""Chunking strategies: Blocks in, Chunks out.

Chunkers never see the source format and never assign IDs (the pipeline does,
after filtering, so IDs are dense and deterministic).
"""
import math
import re
from dataclasses import replace
from itertools import groupby
from typing import Callable, Protocol

from .models import Block, Chunk

LenFn = Callable[[str], int]
TOKENS_PER_WORD = 1.3


def approx_tokens(text: str) -> int:
    """Cheap token estimate. Swap in the embedding model's tokenizer via `len_fn` for exact budgets."""
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


class Chunker(Protocol):
    def chunk(self, blocks: list[Block]) -> list[Chunk]: ...


def blocks_to_chunk(blocks: list[Block], **metadata) -> Chunk:
    parts = [blocks[0].text]
    for prev, cur in zip(blocks, blocks[1:]):
        sep = "\n" if prev.kind == cur.kind == "list_item" else "\n\n"
        parts.append(sep + cur.text)
    return Chunk(
        doc_id=blocks[0].doc_id,
        text="".join(parts),
        heading_path=blocks[0].heading_path,
        pages=sorted({b.page for b in blocks if b.page is not None}),
        kind="table" if all(b.kind == "table" for b in blocks) else "text",
        source_refs=[ref for b in blocks for ref in b.source_refs],
        metadata=metadata,
    )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_text(text: str, max_tokens: int, len_fn: LenFn) -> list[str]:
    """Split text into pieces under max_tokens, on sentence boundaries where possible."""
    units: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len_fn(sentence) <= max_tokens:
            units.append(sentence)
            continue
        words = sentence.split()  # a single run-on "sentence": fall back to word windows
        step = max(1, int(max_tokens / TOKENS_PER_WORD))
        units += [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    pieces, current = [], ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len_fn(candidate) > max_tokens:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


class HybridChunker:
    """Pack consecutive blocks of one section up to max_tokens.

    - never crosses a section (heading_path) boundary
    - splits oversized paragraphs on sentence boundaries
    - tables become their own chunk (kept whole, even if over budget)
    - a tiny trailing chunk is merged into its predecessor
    """

    def __init__(self, max_tokens: int = 400, min_tokens: int = 50, tables_separate: bool = True,
                 len_fn: LenFn = approx_tokens):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tables_separate = tables_separate
        self.len_fn = len_fn

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        chunks = []
        for _, section in groupby(blocks, key=lambda b: b.heading_path):
            chunks += [blocks_to_chunk(group) for group in self._pack_section(list(section))]
        return chunks

    def _explode(self, blocks: list[Block]) -> list[Block]:
        out = []
        for b in blocks:
            if b.kind == "table" or self.len_fn(b.text) <= self.max_tokens:
                out.append(b)
            else:
                out += [replace(b, text=piece) for piece in split_text(b.text, self.max_tokens, self.len_fn)]
        return out

    def _pack_section(self, blocks: list[Block]) -> list[list[Block]]:
        groups: list[list[Block]] = []
        current: list[Block] = []
        current_len = 0

        def flush():
            nonlocal current, current_len
            if current:
                groups.append(current)
            current, current_len = [], 0

        for b in self._explode(blocks):
            if b.kind == "table" and self.tables_separate:
                flush()
                groups.append([b])
                continue
            n = self.len_fn(b.text)
            if current and current_len + n > self.max_tokens:
                flush()
            current.append(b)
            current_len += n
        flush()

        if len(groups) >= 2 and not self._is_table(groups[-1]) and not self._is_table(groups[-2]):
            tail, prev = self._len(groups[-1]), self._len(groups[-2])
            if tail < self.min_tokens and tail + prev <= self.max_tokens + self.min_tokens:
                groups[-2:] = [groups[-2] + groups[-1]]
        return groups

    def _len(self, group: list[Block]) -> int:
        return sum(self.len_fn(b.text) for b in group)

    def _is_table(self, group: list[Block]) -> bool:
        return self.tables_separate and group[0].kind == "table"


class SectionChunker(HybridChunker):
    """One chunk per section (tables inline), split only when a section exceeds max_tokens."""

    def __init__(self, max_tokens: int = 1200, len_fn: LenFn = approx_tokens):
        super().__init__(max_tokens=max_tokens, min_tokens=0, tables_separate=False, len_fn=len_fn)


class FixedSizeChunker:
    """Section-blind sliding word window: the baseline to beat.

    Budgets are converted to words with TOKENS_PER_WORD, so `len_fn` is not used here.
    """

    def __init__(self, max_tokens: int = 300, overlap: int = 50):
        self.window = max(1, int(max_tokens / TOKENS_PER_WORD))
        self.stride = max(1, self.window - int(overlap / TOKENS_PER_WORD))

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        words: list[tuple[str, int]] = [(w, i) for i, b in enumerate(blocks) for w in b.text.split()]
        chunks = []
        for start in range(0, len(words), self.stride):
            window = words[start:start + self.window]
            covered = [blocks[i] for i in sorted({i for _, i in window})]
            chunk = blocks_to_chunk(covered)
            chunk.text = " ".join(w for w, _ in window)
            chunk.embed_text = chunk.text
            chunk.kind = "text"
            chunks.append(chunk)
            if start + self.window >= len(words):
                break
        return chunks
