#!/usr/bin/env python3
"""Compare retrievers with and without a cross-encoder reranker on the top candidates.

Usage:
    uv run --with sentence-transformers python scripts/eval_rerank.py
        [--strategy hybrid] [--rerankers BAAI/bge-reranker-large ...] [--candidates 20 50]
        [--rrf-k 60] [--dense-weights 1] [--with-bm25]

Bases are BM25 + Qwen dense fused with RRF, one per (dense weight, k_rrf) pair, plus
BM25 alone with --with-bm25. Each base is scored without reranking and then with every
(reranker, candidates) pair. Qwen rankings come from the cache written by
scripts/sweep_rrf.py, so after that has run once this script is fully offline (the
rerankers run locally). Reranker scores are cached too, so adding rows is cheap.
OpenScholar/OpenScholar_Reranker also needs `--with sentencepiece`.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.chunking import REGISTRY, build_pipeline
from wattbot.eval import (BM25Retriever, CachedRetriever, ChromaRetriever, CrossEncoderReranker, FusionRetriever,
                          evaluate, load_questions)

PARSED_DIR = Path("documents/parsed_json")
EVAL_OUT = Path("documents/eval")
COLUMNS = ["doc_recall@1", "doc_recall@5", "passage_recall@1", "passage_recall@5", "passage_recall@10",
           "passage_recall@20", "passage_mrr", "ctx_tokens@5"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="hybrid", choices=sorted(REGISTRY))
    parser.add_argument("--rerankers", nargs="+", default=["BAAI/bge-reranker-large"])
    parser.add_argument("--candidates", nargs="+", type=int, default=[20])
    parser.add_argument("--rrf-k", nargs="+", type=int, default=[60])
    parser.add_argument("--dense-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--with-bm25", action="store_true", help="also score BM25 alone as a base")
    args = parser.parse_args()

    questions = load_questions()
    chunks = build_pipeline(args.strategy).run(sorted(PARSED_DIR.glob("*.json")))
    dense_inner = ChromaRetriever(args.strategy)
    dense = CachedRetriever(dense_inner, EVAL_OUT / "cache" / f"{args.strategy}__{dense_inner.name}.json")

    bases = [(f"rrf bm25+qwen w={w:g} k={k}",
              lambda w=w, k=k: FusionRetriever([BM25Retriever(), dense], k_rrf=k, weights=[1.0, w]))
             for k in args.rrf_k for w in args.dense_weights]
    if args.with_bm25:
        bases.insert(0, ("bm25", BM25Retriever))

    print(f"{'config':<58}" + "".join(f"{c.replace('_recall', '').replace('passage', 'p'):>9}" for c in COLUMNS))
    for label, make in bases:
        runs = [(label, make())]
        runs += [(f"{label} + {model.split('/')[-1]} top{n}", CrossEncoderReranker(make(), model, candidates=n))
                 for model in args.rerankers for n in args.candidates]
        for name, retriever in runs:
            summary, _ = evaluate(chunks, questions, retriever)
            print(f"{name:<58}" + "".join(
                f"{summary[c]:>9.0f}" if c.startswith("ctx") else f"{summary[c]:>9.3f}" for c in COLUMNS), flush=True)
    print(f"\nquestions={summary['questions']} (passage: {summary['passage_questions']})")


if __name__ == "__main__":
    main()
