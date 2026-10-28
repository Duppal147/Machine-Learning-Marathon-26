"""Retrieval scoring for the chart questions (`Figure` = 1 in train_QA.csv).

The passage metrics in retrieval.py locate evidence by matching quoted text, which charts don't have,
so those questions are excluded from them. Two label-free checks work instead, both restricted to
chunks of a gold document:

- figure_recall@k  a top-k chunk is the caption of a figure the question's supporting_materials names
                   ("Figure 3"); scored only for questions that name one
- value_recall@k   a top-k chunk contains the answer's number; scored only for numeric answers with a unit

value_recall is the one that measures "exact details": a caption-only index scores low on it because the
number lives in the chart, and an index with the chart's data table can score high. A chunk that merely
repeats the number somewhere unrelated also counts, so read it next to figure_recall.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..chunking import Chunk
from .retrieval import TRAIN_QA, Retriever, _parse_ref_ids

_FIGURE_REF = re.compile(r"\bfig(?:ure)?s?\.?\s*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class FigureQuestion:
    id: str
    question: str
    ref_ids: list[str]
    figure_numbers: set[str] = field(default_factory=set)  # from supporting_materials, e.g. {"3"}
    answer_number: float | None = None


def figure_numbers(text: str) -> set[str]:
    return set(_FIGURE_REF.findall(text))


def numbers_in(text: str) -> set[float]:
    found = set()
    for match in _NUMBER.findall(text):
        try:
            found.add(float(match.replace(",", "")))
        except ValueError:
            pass
    return found


def answer_number(answer_value: str, answer_unit: str) -> float | None:
    """The numeric answer, or None for text, true/false (unit is_blank) and ranges."""
    if (answer_unit or "").strip() in ("", "is_blank"):
        return None
    try:
        return float(answer_value.replace(",", "").strip())
    except ValueError:
        return None


def load_figure_questions(path: Path = TRAIN_QA) -> list[FigureQuestion]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("Figure") == "1" and r.get("is_NA") != "1"]
    return [
        FigureQuestion(
            id=r["id"], question=r["question"], ref_ids=_parse_ref_ids(r["ref_id"]),
            figure_numbers=figure_numbers(r.get("supporting_materials", "")),
            answer_number=answer_number(r.get("answer_value", ""), r.get("answer_unit", "")),
        )
        for r in rows
    ]


def is_caption_of(chunk: Chunk, numbers: set[str]) -> bool:
    """The chunk has a line starting "Figure N" / "Fig. N" for one of the numbers (a caption, not a mention)."""
    if not numbers:
        return False
    pattern = r"(?im)^\s*(?:figure|fig\.?)\s*(?:" + "|".join(sorted(numbers)) + r")\b"
    return re.search(pattern, chunk.text) is not None


def has_value(chunk: Chunk, value: float) -> bool:
    return any(abs(n - value) <= 1e-9 * max(1.0, abs(value)) for n in numbers_in(chunk.text))


@dataclass
class FigureQuestionResult:
    id: str
    figure_rank: int | None  # None = not retrieved, or the question names no figure
    value_rank: int | None   # None = not retrieved, or the answer isn't a plain number
    scores_figure: bool
    scores_value: bool
    top_chunk_ids: list[str]


def evaluate_figures(chunks: list[Chunk], questions: list[FigureQuestion], retriever: Retriever,
                     ks: tuple[int, ...] = (1, 5, 10)):
    """Score one chunk set on the chart questions. Returns (summary dict, per-question results)."""
    corpus_refs = {c.metadata.get("ref_id") for c in chunks} - {None}
    questions = [q for q in questions if any(r in corpus_refs for r in q.ref_ids)]
    if not questions:
        raise ValueError("no chart questions reference documents in this corpus (is ref_id metadata attached?)")

    retriever.fit(chunks)
    ranked = retriever.search([q.question for q in questions], max(ks))

    results = []
    for q, idxs in zip(questions, ranked):
        top = [chunks[i] for i in idxs]
        gold = [(r, c) for r, c in enumerate(top, 1) if c.metadata.get("ref_id") in q.ref_ids]
        figure_rank = next((r for r, c in gold if is_caption_of(c, q.figure_numbers)), None)
        value_rank = next((r for r, c in gold if has_value(c, q.answer_number)), None) \
            if q.answer_number is not None else None
        results.append(FigureQuestionResult(
            q.id, figure_rank, value_rank, bool(q.figure_numbers), q.answer_number is not None,
            [c.chunk_id for c in top[:5]]))

    summary = {
        "retriever": retriever.name,
        "chunks": len(chunks),
        "figure_chunks": sum(c.kind == "figure" for c in chunks),
        "questions": len(results),
        "figure_questions": sum(r.scores_figure for r in results),
        "value_questions": sum(r.scores_value for r in results),
    }
    for metric, flag in (("figure", "scores_figure"), ("value", "scores_value")):
        scored = [r for r in results if getattr(r, flag)]
        for k in ks:
            summary[f"{metric}_recall@{k}"] = (
                np.mean([getattr(r, f"{metric}_rank") is not None and getattr(r, f"{metric}_rank") <= k
                         for r in scored]) if scored else float("nan"))
    return summary, results
