"""Answering chart questions: find the right figures, read values off their images, check the numbers.

Shared by scripts/ask_figures.py (one question) and scripts/eval_figures_e2e.py (the chart questions in
train_QA.csv). The gateway client is passed in, so everything here runs offline against fakes.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import figures
from .hybrid import Bm25Index, rrf_fuse

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FIGURE_WORD = re.compile(r"\bfig(?:ure)?s?\.?\s*\d+(?:\s*\(\w\))?", re.IGNORECASE)  # "Fig. 8", "Figure 2(d)"


def numbers_in(text: str) -> set[float]:
    found = set()
    for match in _NUMBER.findall(text):
        try:
            found.add(float(match.replace(",", "")))
        except ValueError:
            pass
    return found


def reply_numbers(reply: str) -> list[float]:
    """The numbers in the first line of a reading (where ANSWER_PROMPT asks for the exact value).

    "Fig. 8" style references are removed first so a figure number is never mistaken for a value.
    """
    first = next((line for line in reply.splitlines() if line.strip()), "")
    return sorted(numbers_in(_FIGURE_WORD.sub(" ", first)))


def _decimals(value: float) -> int:
    text = repr(float(value))
    return 0 if "e" in text or "." not in text else len(text.split(".")[1].rstrip("0"))


def numbers_match(reply: str, gold: float, tolerance: float = 0.0) -> bool:
    """True if a number in the reply's first line equals gold at gold's own precision.

    28.5 matches 28.5 and 28.54 but not 28.4; 2007 matches only 2007. A relative `tolerance` (0.01 = 1%)
    additionally accepts near values, for charts where a value can only be estimated off the axis; it is
    off by default because 1% of a year or a count is a wrong answer, not a rounding difference.
    """
    places = _decimals(gold)
    return any(round(n, places) == round(gold, places) or math.isclose(n, gold, rel_tol=tolerance, abs_tol=1e-9)
               for n in reply_numbers(reply))


class FigureSearcher:
    """Hybrid search (BM25 + dense, fused with RRF) over figure chunks only.

    `store` is a wattbot.vectorstore.ChunkVectorStore whose collection holds these chunks; without one
    the search is BM25-only and fully offline. `ref_id` restricts a search to one document.
    """

    def __init__(self, chunks: Sequence, store=None, k_rrf: int = 60, depth: int = 50):
        self.chunks = list(chunks)
        self.store, self.k_rrf, self.depth = store, k_rrf, depth
        self.bm25 = Bm25Index().fit([c.embed_text for c in self.chunks])
        self.position = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.refs = np.array([c.metadata.get("ref_id") for c in self.chunks], dtype=object)

    def search(self, question: str, k: int, ref_id: str | None = None) -> list:
        depth = max(k, self.depth)
        scores = self.bm25.scores([question])[0]
        if ref_id is not None:
            scores = np.where(self.refs == ref_id, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:depth]
        rankings = [[int(i) for i in order if scores[i] > 0]]
        if self.store is not None:
            where = {"kind": "figure"} if ref_id is None else {"$and": [{"kind": "figure"}, {"ref_id": ref_id}]}
            hits = self.store.search(question, k=depth, where=where)
            rankings.append([self.position[h.chunk_id] for h in hits if h.chunk_id in self.position])
        return [self.chunks[i] for i in rrf_fuse(rankings, self.k_rrf, limit=k)]


def read_with_retry(client: Any, model: str, question: str, image: Path, context: str, max_tokens: int,
                    extra_body: dict | None, on_retry: Callable[[int], None] | None = None) -> str:
    """Read one chart; if a reasoning model spends the whole budget thinking, retry once with double."""
    try:
        return figures.answer_from_figure(client, model, question, image, context, max_tokens, extra_body)
    except figures.VisionModelError:
        if on_retry:
            on_retry(max_tokens * 2)
        return figures.answer_from_figure(client, model, question, image, context, max_tokens * 2, extra_body)
