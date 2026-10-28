#!/usr/bin/env python3
"""Sweep RRF fusion settings (dense weight, k_rrf) for BM25 + dense hybrid retrieval.

Usage:
    python scripts/sweep_rrf.py [--strategy hybrid] [--dense-weights 0.2 0.3 0.5 1]
                                [--rrf-k 60] [--no-query-instruction] [--dense-model qwen]

--dense-model qwen (default) uses the Qwen gateway embeddings in Chroma. Any other value is
loaded locally as a sentence-transformers model, e.g. BAAI/bge-base-en-v1.5 (needs
`uv run --with sentence-transformers`, no gateway).

Dense rankings are cached in documents/eval/cache/, so only the first run needs the
gateway (OPENAI_API_KEY + UW VPN); later runs with any weights or k are offline.
BM25-only and dense-only rows are printed as baselines. The summary table also goes
to documents/eval/sweep_rrf__<strategy>__<dense model>.csv.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.chunking import REGISTRY, build_pipeline
from wattbot.embeddings import EmbeddingConfigurationError
from wattbot.eval import (BM25Retriever, CachedRetriever, ChromaRetriever, FusionRetriever,
                          SentenceTransformerRetriever, evaluate, load_questions)

PARSED_DIR = Path("documents/parsed_json")
EVAL_OUT = Path("documents/eval")
COLUMNS = ["doc_recall@1", "doc_recall@5", "passage_recall@1", "passage_recall@5", "passage_recall@10",
           "passage_recall@20", "passage_mrr"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="hybrid", choices=sorted(REGISTRY))
    parser.add_argument("--dense-weights", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                        help="dense weight relative to BM25 (BM25 weight is 1)")
    parser.add_argument("--rrf-k", nargs="+", type=int, default=[60])
    parser.add_argument("--depth", type=int, default=50, help="chunks each retriever contributes to the fusion")
    parser.add_argument("--no-query-instruction", action="store_true",
                        help="embed raw questions (Qwen: no instruction; local models: no query prefix)")
    parser.add_argument("--dense-model", default="qwen",
                        help="qwen (gateway, Chroma) or a sentence-transformers model id")
    args = parser.parse_args()

    questions = load_questions()
    chunks = build_pipeline(args.strategy).run(sorted(PARSED_DIR.glob("*.json")))
    if args.dense_model == "qwen":
        dense_inner = ChromaRetriever(args.strategy, query_instruction=not args.no_query_instruction)
    else:
        prefix = None if args.no_query_instruction else SentenceTransformerRetriever.BGE_QUERY_PREFIX
        dense_inner = SentenceTransformerRetriever(args.dense_model, query_prefix=prefix)
    dense = CachedRetriever(dense_inner, EVAL_OUT / "cache" / f"{args.strategy}__{dense_inner.name}.json")

    configs = [("bm25 only", BM25Retriever()), (f"dense only ({dense_inner.name})", dense)]
    configs += [(f"rrf w_dense={w:g} k={k}", FusionRetriever([BM25Retriever(), dense], k_rrf=k,
                                                          depth=args.depth, weights=[1.0, w]))
                for k in args.rrf_k for w in args.dense_weights]

    rows = []
    for label, retriever in configs:
        summary, _ = evaluate(chunks, questions, retriever)
        rows.append((label, summary))
        print(f"{label:<36}" + "".join(f"{summary[c]:>9.3f}" for c in COLUMNS), flush=True)

    out = EVAL_OUT / f"sweep_rrf__{args.strategy}__{dense_inner.name}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config"] + COLUMNS)
        writer.writerows([label] + [f"{s[c]:.4f}" for c in COLUMNS] for label, s in rows)

    first = rows[0][1]
    print(f"\nquestions={first['questions']} (passage: {first['passage_questions']})  "
          f"columns: " + ", ".join(c.replace("_recall", "").replace("passage_", "p_") for c in COLUMNS))
    best = max(rows[2:], key=lambda r: (r[1]["passage_recall@5"], r[1]["passage_mrr"]), default=None)
    if best:
        print(f"best fusion by passage_recall@5 (then MRR): {best[0]}")
    print(f"summary -> {out}")


if __name__ == "__main__":
    try:
        main()
    except EmbeddingConfigurationError as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")
