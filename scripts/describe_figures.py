#!/usr/bin/env python3
"""Read every chart image with the vision model and cache the result for chunking.

Usage:
    python scripts/describe_figures.py [doc_id ...] [--workers 4] [--limit N] [--force] [--dry-run]
                                       [--model NAME] [--min-pixels 10000] [--max-tokens 8000] [--no-thinking]

Reads documents/figures/<doc_id>/figure_NN.png (written by scripts/parse_pdfs.py) together with the
caption and in-figure labels from documents/parsed_json/<doc_id>.json, and writes the model's
structured reading (chart type, axes, legend, data table, summary) to
documents/figure_text/<doc_id>/figure_NN.json. With no doc_ids, every parsed document is processed.

Already-read figures are skipped (same image, same prompt version), JSON is written atomically and a
failed figure is reported and skipped, so an interrupted or partly failed run resumes cleanly.
Images smaller than --min-pixels (icons, bullets) are never sent. Use --limit 5 for a smoke test and
--dry-run to count the calls without making any. Needs OPENAI_API_KEY and the UW VPN for the gateway.
Pick the model with --model or the VISION_MODEL variable; scripts/check_vision_model.py lists what is served.

Then run `python scripts/chunk_docs.py --strategy hybrid_fig` and `python scripts/vector_index.py index
--strategy hybrid_fig`.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import figures
from wattbot.embeddings import EmbeddingConfigurationError, create_embedding_client

PARSED_DIR = Path("documents/parsed_json")


def plan_document(doc_id: str, args) -> tuple[list[dict], dict]:
    """The figures of one document that still need a reading, and counts of the ones that don't."""
    doc = json.loads((PARSED_DIR / f"{doc_id}.json").read_text())
    todo, counts = [], {"missing_image": 0, "too_small": 0, "cached": 0}
    for picture in doc.get("pictures", []):
        index = figures.picture_index(picture)
        image = figures.figure_image_path(doc_id, index)
        if not image.exists():
            counts["missing_image"] += 1
            continue
        size = figures.image_size(image)
        if size and size[0] * size[1] < args.min_pixels:
            counts["too_small"] += 1
            continue
        out = figures.figure_text_path(doc_id, index)
        if not args.force:
            if figures.record_is_current(figures.load_record(out), figures.file_sha256(image)):
                counts["cached"] += 1
                continue
        todo.append({"doc_id": doc_id, "index": index, "image": image, "out": out,
                     "caption": figures.picture_caption(doc, picture),
                     "labels": figures.picture_labels(doc, picture)})
    return todo, counts


def describe(client, model: str, job: dict, max_tokens: int, extra_body: dict | None) -> None:
    record = figures.describe_image(client, model, job["image"], job["caption"], job["labels"],
                                    max_tokens=max_tokens, extra_body=extra_body)
    figures.save_record(job["out"], record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_ids", nargs="*")
    parser.add_argument("--model", default="", help="vision chat model (default: $VISION_MODEL, else auto-detect)")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("--limit", type=int, help="read at most this many figures (smoke test)")
    parser.add_argument("--min-pixels", type=int, default=10_000, help="skip images with fewer pixels than this")
    parser.add_argument("--max-tokens", type=int, default=figures.MAX_TOKENS,
                        help="reply budget per figure; a reasoning model spends part of it thinking")
    parser.add_argument("--no-thinking", action="store_true",
                        help="ask the gateway to switch the model's reasoning off (faster; may be ignored or rejected)")
    parser.add_argument("--force", action="store_true", help="re-read figures that already have a reading")
    parser.add_argument("--dry-run", action="store_true", help="count the work and exit")
    args = parser.parse_args()

    doc_ids = args.doc_ids or sorted(p.stem for p in PARSED_DIR.glob("*.json"))
    if not doc_ids:
        sys.exit(f"error: no parsed documents in {PARSED_DIR} (run scripts/parse_pdfs.py or fetch_index.py)")

    jobs, totals = [], {"missing_image": 0, "too_small": 0, "cached": 0}
    for doc_id in doc_ids:
        doc_jobs, counts = plan_document(doc_id, args)
        jobs += doc_jobs
        for key, n in counts.items():
            totals[key] += n
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(doc_ids)} documents: {len(jobs)} figures to read; skipped {totals['cached']} already read, "
          f"{totals['too_small']} too small, {totals['missing_image']} without an image file")
    if totals["missing_image"] and not jobs and not totals["cached"]:
        sys.exit("error: no figure images found; run scripts/parse_pdfs.py --all --force to write documents/figures/")
    if args.dry_run or not jobs:
        return

    try:
        client = create_embedding_client()
        model = figures.resolve_vision_model(client, args.model)
    except (EmbeddingConfigurationError, figures.VisionModelError) as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")
    print(f"reading with {model}, {args.workers} workers")

    failed, start = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        extra_body = figures.NO_THINKING if args.no_thinking else None
        futures = {pool.submit(describe, client, model, job, args.max_tokens, extra_body): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                failed.append(f"{job['doc_id']}/figure_{job['index']:02d}")
                print(f"FAIL {failed[-1]}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} ({time.monotonic() - start:.0f}s)", flush=True)

    print(f"done: read={len(jobs) - len(failed)} failed={len(failed)} -> {figures.FIGURE_TEXT_DIR}")
    if failed:
        print("re-run the same command to retry: " + " ".join(failed[:20]) + (" ..." if len(failed) > 20 else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
