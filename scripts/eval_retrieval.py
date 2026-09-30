#!/usr/bin/env python3
"""Compare chunking strategies by retrieval quality on WattBot train questions.

Usage:
    python scripts/eval_retrieval.py [--strategies hybrid hybrid_raw ...] [--retriever tfidf|chroma]
                             [--no-query-instruction]

Chunks are rebuilt from documents/parsed_json for each strategy. Only questions whose
gold documents have been parsed are scored. Per-question results (for failure analysis)
go to documents/eval/<strategy>__<retriever>.csv.

--retriever chroma embeds with the configured Qwen model (needs OPENAI_API_KEY + UW VPN) and
reuses the persisted index in documents/chroma, so only new or changed chunks are embedded.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.chunking import REGISTRY, build_pipeline
from wattbot.embeddings import EmbeddingConfigurationError
from wattbot.eval import ChromaRetriever, TfidfRetriever, evaluate, load_questions

PARSED_DIR = Path("documents/parsed_json")
EVAL_OUT = Path("documents/eval")
COLUMNS = ["doc_recall@1", "doc_recall@5", "passage_recall@1", "passage_recall@5", "passage_recall@10",
           "passage_recall@20", "passage_mrr", "ctx_tokens@5"]


def make_retriever(args, strategy):
    if args.retriever == "tfidf":
        return TfidfRetriever()
    return ChromaRetriever(strategy, query_instruction=not args.no_query_instruction)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=sorted(REGISTRY), choices=sorted(REGISTRY))
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "chroma"])
    parser.add_argument("--no-query-instruction", action="store_true",
                        help="chroma: embed raw questions without the Qwen query instruction")
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    questions = load_questions()
    paths = sorted(PARSED_DIR.glob("*.json"))
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy in args.strategies:
        chunks = build_pipeline(strategy, max_tokens=args.max_tokens).run(paths)
        retriever = make_retriever(args, strategy)
        summary, results = evaluate(chunks, questions, retriever)
        rows.append((strategy, summary))

        out = EVAL_OUT / f"{strategy}__{retriever.name.replace(':', '_').replace('/', '_')}.csv"
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "ref_ids", "has_passage_gold", "doc_rank", "passage_rank", "top5_chunk_ids"])
            for r in results:
                writer.writerow([r.id, " ".join(r.ref_ids), r.has_passage_gold, r.doc_rank, r.passage_rank,
                                 " ".join(r.top_chunk_ids)])

    first = rows[0][1]
    print(f"retriever={first['retriever']}  questions={first['questions']} "
          f"(with verbatim evidence located: {first['passage_questions']})\n")
    print(f"{'strategy':<12}{'chunks':>7}" + "".join(f"{c:>19}" for c in COLUMNS))
    for strategy, s in rows:
        print(f"{strategy:<12}{s['chunks']:>7}" + "".join(
            f"{s[c]:>19.0f}" if c.startswith("ctx") else f"{s[c]:>19.3f}" for c in COLUMNS))
    print(f"\nper-question results -> {EVAL_OUT}/")


if __name__ == "__main__":
    try:
        main()
    except EmbeddingConfigurationError as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")
