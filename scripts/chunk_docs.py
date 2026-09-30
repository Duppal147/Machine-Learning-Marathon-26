#!/usr/bin/env python3
"""Chunk parsed documents into documents/chunks/<strategy>.jsonl.

Usage:
    python scripts/chunk_docs.py [--strategy hybrid] [--max-tokens 400] [doc_id ...]

With no doc_ids, every file in documents/parsed_json is chunked.
Strategies are defined in src/wattbot/chunking/pipeline.py (REGISTRY).
"""
import argparse
import os
import statistics
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.chunking import REGISTRY, ChunkStore, approx_tokens, build_pipeline

PARSED_DIR = Path("documents/parsed_json")
CHUNKS_OUT = Path("documents/chunks")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_ids", nargs="*")
    parser.add_argument("--strategy", default="hybrid", choices=sorted(REGISTRY))
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    if args.doc_ids:
        paths = [PARSED_DIR / f"{doc_id}.json" for doc_id in args.doc_ids]
    else:
        paths = sorted(PARSED_DIR.glob("*.json"))

    pipeline = build_pipeline(args.strategy, max_tokens=args.max_tokens)
    chunks = pipeline.run(paths)
    store = ChunkStore(chunks)

    out_path = CHUNKS_OUT / f"{args.strategy}.jsonl"
    store.save(out_path)

    per_doc = Counter(c.doc_id for c in chunks)
    for doc_id, n in per_doc.items():
        lengths = [approx_tokens(c.text) for c in store.by_doc(doc_id)]
        print(f"{doc_id}: chunks={n} tokens median={statistics.median(lengths):.0f} max={max(lengths)}")
    print(f"{len(store)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
