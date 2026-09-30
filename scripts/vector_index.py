#!/usr/bin/env python3
"""Embed chunks into ChromaDB and search them.

Usage:
    python scripts/vector_index.py index  --strategy hybrid [--chunks PATH] [--prune] [--rebuild]
    python scripts/vector_index.py search --strategy hybrid "question" [-k 5] [--ref-id shehabi2024]

`index` reads documents/chunks/<strategy>.jsonl (from scripts/chunk_docs.py) and only
embeds chunks that are new or whose embed_text changed, so re-running is cheap and an
interrupted run resumes. Needs OPENAI_API_KEY and the UW VPN for the BadgerBrain gateway.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import config
from wattbot.chunking import ChunkStore
from wattbot.embeddings import EmbeddingConfigurationError
from wattbot.vectorstore import ChunkVectorStore

CHUNKS_DIR = Path("documents/chunks")
EMBED_BATCH = 16  # per-request batch inside wattbot.embeddings.embed_texts


def cmd_index(args, store: ChunkVectorStore) -> None:
    chunks_path = Path(args.chunks or CHUNKS_DIR / f"{args.strategy}.jsonl")
    chunks = list(ChunkStore.load(chunks_path))
    if args.rebuild:
        store.reset()

    start = time.monotonic()

    def progress(stats):
        print(f"  embedded {stats.embedded} chunks ({time.monotonic() - start:.0f}s)", flush=True)

    print(f"indexing {len(chunks)} chunks from {chunks_path} into {store.collection.name} ...")
    stats = store.index(chunks, prune=args.prune, progress=progress)
    api_calls = -(-stats.embedded // EMBED_BATCH)
    print(f"done: embedded={stats.embedded} skipped={stats.skipped} deleted={stats.deleted} "
          f"api_calls~{api_calls} elapsed={time.monotonic() - start:.0f}s total_in_store={store.count()}")


def cmd_search(args, store: ChunkVectorStore) -> None:
    where = {"ref_id": args.ref_id} if args.ref_id else None
    instruction = None if args.no_query_instruction else config.QUERY_INSTRUCTION
    for rank, hit in enumerate(store.search(args.question, k=args.k, where=where, instruction=instruction), 1):
        m = hit.metadata
        snippet = " ".join(hit.text.split())[:160]
        print(f"{rank:>2}. {hit.score:.3f}  {m.get('ref_id', m['doc_id'])}  p.{m['pages']}  {m['heading_path']}")
        print(f"      {hit.chunk_id}\n      {snippet}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="embed chunks and upsert them into ChromaDB")
    p_index.add_argument("--strategy", default="hybrid")
    p_index.add_argument("--chunks", help="chunks JSONL (default documents/chunks/<strategy>.jsonl)")
    p_index.add_argument("--prune", action="store_true", help="delete stored chunks no longer in the JSONL")
    p_index.add_argument("--rebuild", action="store_true", help="drop the collection and re-embed everything")

    p_search = sub.add_parser("search", help="search the index with a question")
    p_search.add_argument("question")
    p_search.add_argument("--strategy", default="hybrid")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--ref-id", help="only search one document (WattBot ref_id)")
    p_search.add_argument("--no-query-instruction", action="store_true",
                          help="embed the raw question without the Qwen query instruction")

    args = parser.parse_args()
    store = ChunkVectorStore(args.strategy)
    try:
        {"index": cmd_index, "search": cmd_search}[args.command](args, store)
    except EmbeddingConfigurationError as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")


if __name__ == "__main__":
    main()
