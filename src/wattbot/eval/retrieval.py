"""Retrieval evaluation against WattBot train questions.

For each question we know the gold documents (`ref_id`) and, usually, verbatim
evidence (`supporting_materials`). A chunking strategy is scored by retrieving
chunks for the question text and checking:

- doc_recall@k      a top-k chunk comes from a gold document
- passage_recall@k  a top-k chunk contains the evidence (fuzzy word-sequence match)
- passage_mrr       reciprocal rank of the first evidence-bearing chunk
- ctx_tokens@k      mean tokens in the top-k chunks (bigger chunks buy recall with context budget)

Only questions whose gold documents are in the chunked corpus are scored, so
numbers are comparable across strategies but grow in coverage as more PDFs are parsed.
"""
import ast
import csv
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

import numpy as np

from ..chunking import Chunk
from ..chunking.chunkers import approx_tokens

TRAIN_QA = Path("WattBot2026/train_QA.csv")
PASSAGE_THRESHOLD = 0.7  # fraction of an evidence segment's words that must appear, in order, in a chunk
MIN_SEGMENT_WORDS = 6


# ---------------------------------------------------------------------------
# Questions and evidence
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    question: str
    ref_ids: list[str]
    evidence: list[str]  # verbatim segments extracted from supporting_materials
    flags: dict = field(default_factory=dict)  # Table / Figure / Math / Quote evidence-type flags


def _parse_ref_ids(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if raw.startswith("["):
        try:
            return [str(r) for r in ast.literal_eval(raw)]
        except (ValueError, SyntaxError):
            pass
    return [raw] if raw and raw != "is_blank" else []


_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")
_LABEL = re.compile(r"^\s*(?:quote|evidence)\s*:\s*", re.IGNORECASE)
_REF_MARKER = re.compile(r"\b[a-z]+\d{4}[a-z]?\s*:\s*")  # "luccioni2025c: ..." introduces a per-doc snippet
_SEPARATORS = re.compile(r"\.\.\.|…|\s--\s")


def _segments(candidates: list[str]) -> list[str]:
    parts = [p.strip(" .…'\"") for c in candidates for p in _SEPARATORS.split(c)]
    return [p for p in parts if len(p.split()) >= MIN_SEGMENT_WORDS]


def extract_evidence(supporting: str) -> list[str]:
    """Pull verbatim segments out of supporting_materials.

    Prefers text inside double quotes. If no quoted span is long enough (quotes around a
    short term, or none at all), uses the whole text split at "refid:" markers. Either way
    "...", "…" and " -- " mark elisions or commentary, so segments are split there.
    """
    return (_segments(_QUOTED.findall(supporting))
            or _segments(_REF_MARKER.split(_LABEL.sub("", supporting))))


def load_questions(path: Path = TRAIN_QA) -> list[Question]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if row.get("is_NA") == "1":
            continue
        refs = _parse_ref_ids(row["ref_id"])
        if not refs:
            continue
        out.append(Question(
            id=row["id"],
            question=row["question"],
            ref_ids=refs,
            evidence=extract_evidence(row.get("supporting_materials", "")),
            flags={k: row.get(k) == "1" for k in ("Quote", "Table", "Figure", "Math", "CrossPaper")},
        ))
    return out


_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def evidence_coverage(segment: str, chunk_text: str) -> float:
    """Fraction of the segment's words found in the chunk in the same order (robust to Docling spacing)."""
    seg, chunk = _words(segment), _words(chunk_text)
    if not seg:
        return 0.0
    matcher = SequenceMatcher(None, seg, chunk, autojunk=False)
    return sum(b.size for b in matcher.get_matching_blocks()) / len(seg)


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def search(self, queries: list[str], k: int) -> list[list[int]]: ...


class TfidfRetriever:
    """Offline lexical baseline (no API needed)."""

    name = "tfidf"

    def fit(self, texts: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, queries: list[str], k: int) -> list[list[int]]:
        scores = (self.vectorizer.transform(queries) @ self.matrix.T).toarray()
        return [list(np.argsort(-row)[:k]) for row in scores]


class OpenAIEmbeddingRetriever:
    """Dense retrieval through any OpenAI-compatible /embeddings endpoint (e.g. the UW gateway).

    Reads OPENAI_API_KEY from the environment. Embeddings are recomputed each run.
    """

    def __init__(self, model: str, base_url: str | None = None, batch_size: int = 64):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"), timeout=300)
        self.model = model
        self.batch_size = batch_size
        self.name = f"embed:{model}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            vecs += [d.embedding for d in resp.data]
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    def fit(self, texts: list[str]) -> None:
        self.matrix = self._embed(texts)

    def search(self, queries: list[str], k: int) -> list[list[int]]:
        scores = self._embed(queries) @ self.matrix.T
        return [list(np.argsort(-row)[:k]) for row in scores]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: str
    ref_ids: list[str]
    has_passage_gold: bool
    doc_rank: int | None       # 1-based rank of first chunk from a gold doc
    passage_rank: int | None   # 1-based rank of first chunk containing the evidence
    top_chunk_ids: list[str]


def evaluate(chunks: list[Chunk], questions: list[Question], retriever: Retriever,
             ks: tuple[int, ...] = (1, 5, 10, 20), threshold: float = PASSAGE_THRESHOLD):
    """Score one chunk set. Returns (summary dict, per-question results)."""
    corpus_refs = {c.metadata.get("ref_id") for c in chunks} - {None}
    questions = [q for q in questions if any(r in corpus_refs for r in q.ref_ids)]
    if not questions:
        raise ValueError("no questions reference documents in this corpus (is ref_id metadata attached?)")

    retriever.fit([c.embed_text for c in chunks])
    max_k = max(ks)
    ranked = retriever.search([q.question for q in questions], max_k)

    def has_evidence(q: Question, chunk: Chunk) -> bool:
        return chunk.metadata.get("ref_id") in q.ref_ids and any(
            evidence_coverage(seg, chunk.text) >= threshold for seg in q.evidence)

    results = []
    for q, idxs in zip(questions, ranked):
        top = [chunks[i] for i in idxs]
        # passage gold exists only if some chunk in a gold doc contains the evidence (paraphrased
        # or figure-only evidence has none), so passage metrics are computed over that subset
        gold_doc_chunks = [c for c in chunks if c.metadata.get("ref_id") in q.ref_ids]
        has_gold = any(has_evidence(q, c) for c in gold_doc_chunks)
        doc_rank = next((r for r, c in enumerate(top, 1) if c.metadata.get("ref_id") in q.ref_ids), None)
        passage_rank = next((r for r, c in enumerate(top, 1) if has_evidence(q, c)), None) if has_gold else None
        results.append(QuestionResult(q.id, q.ref_ids, has_gold, doc_rank, passage_rank,
                                      [c.chunk_id for c in top[:5]]))

    with_gold = [r for r in results if r.has_passage_gold]
    summary = {
        "retriever": retriever.name,
        "chunks": len(chunks),
        "questions": len(results),
        "passage_questions": len(with_gold),
    }
    for k in ks:
        summary[f"doc_recall@{k}"] = np.mean([r.doc_rank is not None and r.doc_rank <= k for r in results])
    for k in ks:
        summary[f"passage_recall@{k}"] = (
            np.mean([r.passage_rank is not None and r.passage_rank <= k for r in with_gold]) if with_gold else float("nan"))
    summary["passage_mrr"] = (
        np.mean([1 / r.passage_rank if r.passage_rank else 0.0 for r in with_gold]) if with_gold else float("nan"))
    for k in (5,):
        summary[f"ctx_tokens@{k}"] = np.mean(
            [sum(approx_tokens(chunks[i].text) for i in idxs[:k]) for idxs in ranked])
    return summary, results
