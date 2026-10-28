#!/usr/bin/env python3
"""Compare how well each figure strategy retrieves the chart questions (Figure = 1 in train_QA.csv).

Usage:
    python scripts/eval_figures.py [--strategies hybrid hybrid_figlabels hybrid_fig]
                                   [--retriever bm25|tfidf|chroma|hybrid] [--rrf-k 60]

Strategies:
    hybrid            figure captions only (what the index has today)
    hybrid_figlabels  + a figure chunk per chart from the labels in the PDF text layer (no model)
    hybrid_fig        + a figure chunk per chart from the vision-model readings (scripts/describe_figures.py)

Chunks are rebuilt from documents/parsed_json. The metrics (figure_recall@k, value_recall@k) are
explained in src/wattbot/eval/figures.py; bm25 and tfidf run offline, chroma and hybrid need the gateway
and index any missing chunks first. Per-question results go to documents/eval/figures__<strategy>__<retriever>.csv.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.chunking import REGISTRY, build_pipeline
from wattbot.embeddings import EmbeddingConfigurationError
from wattbot.eval import BM25Retriever, ChromaRetriever, FusionRetriever, TfidfRetriever, evaluate_figures
from wattbot.eval import load_figure_questions

PARSED_DIR = Path("documents/parsed_json")
EVAL_OUT = Path("documents/eval")
COLUMNS = ["figure_chunks", "figure_recall@1", "figure_recall@5", "figure_recall@10",
           "value_recall@1", "value_recall@5", "value_recall@10"]


def make_retriever(args, strategy):
    if args.retriever == "bm25":
        return BM25Retriever()
    if args.retriever == "tfidf":
        return TfidfRetriever()
    dense = ChromaRetriever(strategy)
    if args.retriever == "hybrid":
        return FusionRetriever([BM25Retriever(), dense], k_rrf=args.rrf_k)
    return dense


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=["hybrid", "hybrid_figlabels", "hybrid_fig"],
                        choices=sorted(REGISTRY))
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf", "chroma", "hybrid"])
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()

    questions = load_figure_questions()
    paths = sorted(PARSED_DIR.glob("*.json"))
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        for strategy in args.strategies:
            chunks = build_pipeline(strategy).run(paths)
            retriever = make_retriever(args, strategy)
            summary, results = evaluate_figures(chunks, questions, retriever)
            rows.append((strategy, summary))

            out = EVAL_OUT / f"figures__{strategy}__{retriever.name.replace(':', '_').replace('/', '_')}.csv"
            with open(out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "figure_rank", "value_rank", "top5_chunk_ids"])
                for r in results:
                    writer.writerow([r.id, r.figure_rank or "", r.value_rank or "", " ".join(r.top_chunk_ids)])
    except EmbeddingConfigurationError as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")

    s = rows[0][1]
    print(f"retriever={s['retriever']}  chart questions={s['questions']}  "
          f"naming a figure={s['figure_questions']}  numeric answer={s['value_questions']}")
    print(f"{'strategy':<18}" + "".join(f"{c:>17}" for c in COLUMNS))
    for strategy, summary in rows:
        print(f"{strategy:<18}" + "".join(f"{summary[c]:>17.3f}" if c != "figure_chunks" else f"{summary[c]:>17}"
                                          for c in COLUMNS))


if __name__ == "__main__":
    main()
