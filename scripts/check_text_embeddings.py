#!/usr/bin/env python3
"""Run live validation for qwen3-vl-embedding-8b text embeddings."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import config
from wattbot.embeddings import (
    cosine_similarity,
    embed_text,
    embed_texts,
    embedding_l2_norm,
)

DEFAULT_TEXT = "Data centers consumed significant electricity in 2023."
QUERY = "How much electricity did data centers consume in 2023?"
PASSAGES = {
    "relevant": "A report estimates the total electricity used by data centers during 2023.",
    "somewhat_related": "Training large AI models requires energy-intensive computing hardware.",
    "unrelated": "Wetland restoration provides habitat for migratory birds.",
}
STABILITY_THRESHOLD = 0.9999


def _vector_status(vector: Sequence[float]) -> tuple[bool, bool, float]:
    correct_dimension = len(vector) == config.EMBEDDING_DIM
    all_finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    )
    norm = embedding_l2_norm(vector) if all_finite else float("nan")
    return correct_dimension, all_finite, norm


def run_diagnostic(
    text: str,
    *,
    embed_many: Callable[[list[str]], list[list[float]]] = embed_texts,
    embed_one: Callable[[str], list[float]] = embed_text,
    stream=sys.stdout,
) -> int:
    labels = ["arbitrary_text", "query", *PASSAGES]
    vectors = embed_many([text, QUERY, *PASSAGES.values()])
    if len(vectors) != len(labels):
        print(
            f"FAIL: received {len(vectors)} vectors for {len(labels)} inputs",
            file=stream,
        )
        return 1

    print("Qwen text embedding diagnostic", file=stream)
    print(f"model: {config.EMBEDDING_MODEL}", file=stream)
    print(f"expected dimension: {config.EMBEDDING_DIM}", file=stream)
    print("", file=stream)

    checks_passed = True
    for label, vector in zip(labels, vectors):
        correct_dimension, all_finite, norm = _vector_status(vector)
        checks_passed &= correct_dimension and all_finite and norm > 0.0
        print(
            f"{label}: dimension={len(vector)} finite={all_finite} "
            f"l2_norm={norm:.9f}",
            file=stream,
        )

    repeated = embed_one(text)
    repeated_dimension, repeated_finite, repeated_norm = _vector_status(repeated)
    checks_passed &= repeated_dimension and repeated_finite and repeated_norm > 0.0

    stability = cosine_similarity(vectors[0], repeated)
    max_absolute_difference = max(
        abs(first - second)
        for first, second in zip(vectors[0], repeated)
    )
    stability_passed = stability >= STABILITY_THRESHOLD
    checks_passed &= stability_passed
    print("", file=stream)
    print("repeat-call stability:", file=stream)
    print(f"  cosine_similarity={stability:.9f}", file=stream)
    print(f"  max_absolute_difference={max_absolute_difference:.9g}", file=stream)
    print(f"  passed={stability_passed}", file=stream)

    query_vector = vectors[1]
    similarities = {
        label: cosine_similarity(query_vector, vectors[index])
        for index, label in enumerate(PASSAGES, start=2)
    }
    semantic_order_passed = (
        similarities["relevant"]
        > similarities["somewhat_related"]
        > similarities["unrelated"]
    )
    checks_passed &= semantic_order_passed

    print("", file=stream)
    print("WattBot-style semantic similarities:", file=stream)
    for label, score in sorted(
        similarities.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"  {label:18s} {score:.9f}", file=stream)
    print(f"  expected_order_passed={semantic_order_passed}", file=stream)
    print("", file=stream)
    print(f"overall: {'PASS' if checks_passed else 'FAIL'}", file=stream)
    return 0 if checks_passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="arbitrary text to embed")
    args = parser.parse_args(argv)

    try:
        return run_diagnostic(args.text)
    except Exception as error:
        print(f"Diagnostic failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
