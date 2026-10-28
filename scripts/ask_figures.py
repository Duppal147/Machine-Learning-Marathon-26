#!/usr/bin/env python3
"""Find the chart that answers a question, then read the exact value off its image.

Usage:
    python scripts/ask_figures.py "question" [--strategy hybrid_fig] [-k 5] [--read 1]
                                  [--ref-id shehabi2024] [--no-dense] [--model NAME]

Searches only the figure chunks of documents/chunks/<strategy>.jsonl: BM25 over their text fused with
dense Qwen retrieval (RRF, as in docs/hybrid_retrieval_design.md). The top --read figures are then sent
back to the vision model as images together with the question, which is more exact than the reading
cached at indexing time. --no-dense (or the absence of a Chroma index) makes it BM25-only and offline
for the search step; the final read always needs the gateway.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import config, figures
from wattbot.chunking import ChunkStore
from wattbot.embeddings import EmbeddingConfigurationError, create_embedding_client
from wattbot.hybrid import Bm25Index, rrf_fuse

CHUNKS_DIR = Path("documents/chunks")


def search_figures(question: str, chunks, k: int, strategy: str, dense: bool) -> list:
    """Rank figure chunks for a question; chunks is the (already filtered) list to search."""
    rankings = [Bm25Index().fit([c.embed_text for c in chunks]).search([question], max(k, 50))[0]]
    if dense:
        from wattbot.vectorstore import ChunkVectorStore
        store = ChunkVectorStore(strategy)
        position = {c.chunk_id: i for i, c in enumerate(chunks)}
        hits = store.search(question, k=max(k, 50), where={"kind": "figure"})
        rankings.append([position[h.chunk_id] for h in hits if h.chunk_id in position])
    return [chunks[i] for i in rrf_fuse(rankings, 60, limit=k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--strategy", default="hybrid_fig")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--read", type=int, default=1, help="send this many top figures to the vision model (0 = search only)")
    parser.add_argument("--ref-id", help="only search one document (WattBot ref_id)")
    parser.add_argument("--no-dense", action="store_true", help="BM25 only")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    chunks_path = CHUNKS_DIR / f"{args.strategy}.jsonl"
    if not chunks_path.exists():
        sys.exit(f"error: {chunks_path} not found; run scripts/chunk_docs.py --strategy {args.strategy}")
    chunks = [c for c in ChunkStore.load(chunks_path)
              if c.kind == "figure" and (not args.ref_id or c.metadata.get("ref_id") == args.ref_id)]
    if not chunks:
        sys.exit("error: no figure chunks; run scripts/describe_figures.py, then scripts/chunk_docs.py "
                 f"--strategy {args.strategy}")

    try:
        top = search_figures(args.question, chunks, args.k, args.strategy, dense=not args.no_dense)
        for rank, c in enumerate(top, 1):
            caption = " ".join(c.text.split("\n")[1:2])[:110]
            print(f"{rank:>2}. {c.metadata.get('ref_id', c.doc_id)}  p.{c.pages}  figure_{c.metadata['figure_index']:02d}  {caption}")
        if not args.read:
            return

        client = create_embedding_client()
        model = figures.resolve_vision_model(client, args.model)
        for c in top[:args.read]:
            image = figures.figure_image_path(c.doc_id, c.metadata["figure_index"])
            if not image.exists():
                print(f"\n{c.chunk_id}: image {image} not found (run scripts/parse_pdfs.py)")
                continue
            print(f"\n--- {c.metadata.get('ref_id', c.doc_id)} {image.name} (read with {model}) ---")
            print(figures.answer_from_figure(client, model, args.question, image, context=c.text))
    except (EmbeddingConfigurationError, figures.VisionModelError) as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")


if __name__ == "__main__":
    main()
