#!/usr/bin/env python3
"""Find the chart that answers a question, then read the exact value off its image.

Usage:
    python scripts/ask_figures.py "question" [--strategy hybrid_fig] [-k 5] [--read 1]
                                  [--ref-id shehabi2024] [--no-dense] [--model NAME]
                                  [--max-tokens 8000] [--no-thinking] [--prompt default|scope]

Searches only the figure chunks of documents/chunks/<strategy>.jsonl: BM25 over their text fused with
dense Qwen retrieval (RRF, as in docs/hybrid_retrieval_design.md). The top --read figures are then sent
back to the vision model as images together with the question, which is more exact than the reading
cached at indexing time. --no-dense makes the search BM25-only and offline; the final read always
needs the gateway. A read that runs out of tokens is retried once at double the budget.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import figures
from wattbot.chunking import ChunkStore
from wattbot.embeddings import EmbeddingConfigurationError, create_embedding_client
from wattbot.figure_qa import FigureSearcher, read_with_retry

CHUNKS_DIR = Path("documents/chunks")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--strategy", default="hybrid_fig")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--read", type=int, default=1, help="send this many top figures to the vision model (0 = search only)")
    parser.add_argument("--ref-id", help="only search one document (WattBot ref_id)")
    parser.add_argument("--no-dense", action="store_true", help="BM25 only")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=figures.MAX_TOKENS,
                        help="reply budget per figure (retried once at double if the model runs out)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="ask the gateway to switch the model's reasoning off (faster; may be ignored or rejected)")
    parser.add_argument("--prompt", choices=sorted(figures.ANSWER_PROMPTS), default="default",
                        help="'scope' makes the model reject charts on a related but different subject")
    args = parser.parse_args()

    chunks_path = CHUNKS_DIR / f"{args.strategy}.jsonl"
    if not chunks_path.exists():
        sys.exit(f"error: {chunks_path} not found; run scripts/chunk_docs.py --strategy {args.strategy}")
    chunks = [c for c in ChunkStore.load(chunks_path) if c.kind == "figure"]
    if not chunks:
        sys.exit("error: no figure chunks; run scripts/describe_figures.py, then scripts/chunk_docs.py "
                 f"--strategy {args.strategy}")

    try:
        store = None
        if not args.no_dense:
            from wattbot.vectorstore import ChunkVectorStore
            store = ChunkVectorStore(args.strategy)
        top = FigureSearcher(chunks, store).search(args.question, args.k, ref_id=args.ref_id)
        for rank, c in enumerate(top, 1):
            caption = " ".join(c.text.split("\n")[1:2])[:110]
            print(f"{rank:>2}. {c.metadata.get('ref_id', c.doc_id)}  p.{c.pages}  figure_{c.metadata['figure_index']:02d}  {caption}")
        if not args.read:
            return

        client = create_embedding_client()
        model = figures.resolve_vision_model(client, args.model)
        extra_body = figures.NO_THINKING if args.no_thinking else None
        for c in top[:args.read]:
            image = figures.figure_image_path(c.doc_id, c.metadata["figure_index"])
            if not image.exists():
                print(f"\n{c.chunk_id}: image {image} not found (run scripts/parse_pdfs.py)")
                continue
            print(f"\n--- {c.metadata.get('ref_id', c.doc_id)} {image.name} (read with {model}) ---")
            try:
                print(read_with_retry(client, model, args.question, image, c.text, args.max_tokens, extra_body,
                                      prompt=args.prompt, on_retry=lambda n: print(f"  no answer in the first budget; retrying with {n}",
                                                               flush=True)))
            except figures.VisionModelError as error:  # one hard chart shouldn't stop the others
                print(f"error: {error}")
    except (EmbeddingConfigurationError, figures.VisionModelError) as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")


if __name__ == "__main__":
    main()
