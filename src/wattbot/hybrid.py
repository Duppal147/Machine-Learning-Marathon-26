"""Lexical BM25 scoring and Reciprocal Rank Fusion for hybrid retrieval.

Dense retrieval matches meaning but misses exact numbers, acronyms and model names;
BM25 matches those terms but not paraphrases. On the 122-document corpus the two find
the evidence for different questions, so fusing their rankings recovers both.

RRF fuses by rank rather than score, because BM25 and cosine scores live on different
scales and would need calibration to be added together.

Pure numpy/scipy/scikit-learn, so this works offline and adds no dependencies.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")  # keeps "1.9", "4,096", "h100" intact


def tokenize(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class Bm25Index:
    """Okapi BM25 over a fixed set of texts.

    Each document's term weights are precomputed into one sparse matrix, so scoring a
    batch of queries is a single sparse matrix product.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, texts: Sequence[str]) -> "Bm25Index":
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float64)  # docs x terms

        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        # Lucene's idf variant, which stays positive even for terms in most documents
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if n_docs else 0.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / (avg_len or 1.0))

        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = self.idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm[rows])
        self.weights = tf
        return self

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        """BM25 score of every document for every query: shape (queries, docs)."""
        q = self.vectorizer.transform(queries)
        return (q @ self.weights.T).toarray()

    def search(self, queries: Sequence[str], k: int) -> list[list[int]]:
        """Top-k document indices per query, best first.

        Documents sharing no term with the query (score 0) are left out: their order would
        be arbitrary, and in a fusion they would add rank credit backed by no evidence.
        """
        results = []
        for row in self.scores(queries):
            top = np.argsort(-row, kind="stable")[:k]
            results.append([int(i) for i in top if row[i] > 0])
        return results


def rrf_fuse(rankings: Sequence[Sequence[int]], k_rrf: int = 60, limit: int | None = None,
             weights: Sequence[float] | None = None) -> list[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of weight / (k_rrf + rank of d).

    `rankings` are lists of item ids, best first; an item missing from a ranking simply
    gets nothing from it. `weights` (default all 1) let a stronger retriever dominate the
    top ranks while a weaker one still fills gaps further down. Ties keep first-seen
    order, so the result is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + weight / (k_rrf + rank)
    fused = sorted(scores, key=scores.__getitem__, reverse=True)
    return fused[:limit] if limit is not None else fused
