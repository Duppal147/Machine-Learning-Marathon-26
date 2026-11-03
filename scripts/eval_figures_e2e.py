#!/usr/bin/env python3
"""End-to-end check on the chart questions: retrieve charts, read them with the vision model, compare numbers.

Usage:
    python scripts/eval_figures_e2e.py [--strategy hybrid_fig] [--reads 3] [--tolerance 0]
                                       [--oracle-doc] [--no-dense] [--limit N] [--workers 3]
                                       [--model NAME] [--max-tokens 8000] [--no-thinking] [--prompt default [scope ...]]
                                       [--no-cache]

For each Figure-flagged train question with a plain numeric answer, finds the best figure chunks (BM25 +
dense over all figure chunks, as scripts/ask_figures.py does), sends the top --reads chart images with the
question to the vision model, and counts the question as answered when a number on the first line of a
reply equals answer_value at its precision (28.5 matches 28.54, not 28.4; 2007 matches only 2007). Pass
--tolerance 0.01 to also accept values within 1%, for charts that only allow an estimate.

    answer@k          one of the first k charts read gave the right number
    strict_answer@k   ... and that chart is from a gold document, so a coincidence in an unrelated chart doesn't count
    gold_doc_read@k   one of the first k charts read is from a gold document (retrieval alone)
    wrong_number@1    the first chart read gave a number, and it was wrong (the costly failure)

--oracle-doc searches only the gold document, which separates reading errors from retrieval errors.
--prompt takes several names (e.g. `--prompt default scope`): retrieval runs once and every prompt reads the
same charts, so the comparison cannot be confounded by retrieval varying between runs.
Replies are cached in documents/eval/figures_e2e_cache.json, so a rerun only pays for new reads, and
query embeddings in documents/eval/query_embeddings.json, so a rerun retrieves the same charts (the gateway's
vectors for one text differ slightly between calls, enough to reorder close results; --fresh-queries opts out). The
per-question detail goes to documents/eval/figures_e2e__<strategy>.csv: read it, because a unit or rounding
difference from answer_value shows up there as a "wrong" number. About 3 gateway calls per question.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import figures
from wattbot.chunking import ChunkStore
from wattbot.embeddings import CachedEmbedder, EmbeddingConfigurationError, create_embedding_client
from wattbot.eval import evaluate_e2e, load_figure_questions
from wattbot.eval.summary import save_summary
from wattbot.figure_qa import FigureSearcher, read_with_retry

CHUNKS_DIR = Path("documents/chunks")
EVAL_OUT = Path("documents/eval")
CACHE_PATH = EVAL_OUT / "figures_e2e_cache.json"
QUERY_CACHE_PATH = EVAL_OUT / "query_embeddings.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="hybrid_fig")
    parser.add_argument("--reads", type=int, default=3, help="charts read per question")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="extra relative tolerance for a number to match (0.01 = 1%%); default is exact at gold precision")
    parser.add_argument("--oracle-doc", action="store_true", help="search only the gold document")
    parser.add_argument("--no-dense", action="store_true", help="BM25-only retrieval")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--workers", type=int, default=3, help="parallel chart reads")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=figures.MAX_TOKENS)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--prompt", nargs="+", choices=sorted(figures.ANSWER_PROMPTS), default=["default"],
                        help="answer prompt(s); 'scope' makes the model reject charts on a related but different "
                             "subject. Several are compared on the same retrieved charts")
    parser.add_argument("--no-cache", action="store_true", help="ignore and don't write the reply cache")
    parser.add_argument("--fresh-queries", action="store_true",
                        help="embed the questions again instead of reusing the cached query vectors (results may vary "
                             "slightly between runs)")
    args = parser.parse_args()

    chunks_path = CHUNKS_DIR / f"{args.strategy}.jsonl"
    if not chunks_path.exists():
        sys.exit(f"error: {chunks_path} not found; run scripts/chunk_docs.py --strategy {args.strategy}")
    all_chunks = list(ChunkStore.load(chunks_path))
    chunks = [c for c in all_chunks if c.kind == "figure"]
    if not chunks:
        sys.exit("error: no figure chunks in that strategy; it needs hybrid_fig or hybrid_figlabels")
    # every numeric question about a parsed document counts, so a strategy with no figure chunk for the
    # right document is scored as a miss instead of dropping out of the denominator
    corpus_refs = {c.metadata.get("ref_id") for c in all_chunks}
    questions = [q for q in load_figure_questions()
                 if q.answer_number is not None and any(r in corpus_refs for r in q.ref_ids)]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        sys.exit("error: no numeric chart questions for documents in this corpus")

    try:
        store = None
        if not args.no_dense:
            from wattbot.vectorstore import ChunkVectorStore
            # cached query vectors make repeated runs retrieve the same charts (see CachedEmbedder)
            store = ChunkVectorStore(args.strategy, embed_fn=None if args.fresh_queries
                                     else CachedEmbedder(QUERY_CACHE_PATH))
        searcher = FigureSearcher(chunks, store)
        client = create_embedding_client()
        model = figures.resolve_vision_model(client, args.model)
    except (EmbeddingConfigurationError, figures.VisionModelError) as error:
        sys.exit(f"error: {error} (see README: BadgerBrain gateway setup)")
    extra_body = figures.NO_THINKING if args.no_thinking else None

    retrieved: dict[str, list] = {}

    def retrieve(q):
        if q.id not in retrieved:
            ref = q.ref_ids[0] if args.oracle_doc and q.ref_ids else None
            retrieved[q.id] = searcher.search(q.question, args.reads, ref_id=ref)
        return retrieved[q.id]

    # the prompt text is part of the cache key, so editing a prompt never serves replies written for the old one
    prompt_ids = {name: f"{name}-{hashlib.sha256(figures.ANSWER_PROMPTS[name].encode()).hexdigest()[:8]}"
                  for name in args.prompt}
    cache: dict[str, str] = {} if args.no_cache or not CACHE_PATH.exists() else json.loads(CACHE_PATH.read_text())
    failures: dict[str, str] = {}
    lock = threading.Lock()

    def read(q, chunk, prompt):
        key = f"{model}|{int(args.no_thinking)}|{args.max_tokens}|{prompt_ids[prompt]}|{chunk.chunk_id}|{q.question}"
        with lock:
            if key in cache:
                return cache[key]
            if key in failures:
                raise figures.VisionModelError(failures[key])
        image = figures.figure_image_path(chunk.doc_id, chunk.metadata["figure_index"])
        if not image.exists():
            raise FileNotFoundError(f"{image} not found (run scripts/parse_pdfs.py)")
        try:
            reply = read_with_retry(client, model, q.question, image, chunk.text, args.max_tokens, extra_body,
                                    prompt=prompt)
        except figures.VisionModelError as error:
            with lock:
                failures[key] = str(error)
            raise
        with lock:
            cache[key] = reply
        return reply

    # retrieval happens once; every prompt then reads exactly these charts
    jobs = [(q, c, name) for q in questions for c in retrieve(q) for name in args.prompt]
    print(f"{len(questions)} questions, {len(jobs)} chart reads with {model} ({args.workers} workers, "
          f"prompts: {', '.join(args.prompt)})", flush=True)

    def warm(job):
        try:
            read(*job)
        except (figures.VisionModelError, OSError):
            pass  # recorded in `failures`; evaluate_e2e reports it per read

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(warm, jobs))
    if not args.no_cache:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache))

    n = args.reads
    metrics = ["gold_doc_read@1", f"gold_doc_read@{n}", "answer@1", f"answer@{n}", "strict_answer@1",
               f"strict_answer@{n}", "wrong_number@1"]
    summaries = {}
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    for name in args.prompt:
        summary, results = evaluate_e2e(questions, retrieve, lambda q, c, name=name: read(q, c, name),
                                        n_reads=n, tolerance=args.tolerance)
        summaries[name] = summary
        out = EVAL_OUT / (f"figures_e2e__{args.strategy}{'__oracle' if args.oracle_doc else ''}"
                          f"{'' if name == 'default' else '__' + name}.csv")
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gold", "rank", "ref_id", "figure_index", "from_gold_doc", "correct", "numbers",
                             "reply_first_line"])
            for res in results:
                for r in res.reads:
                    first = next((line for line in r.reply.splitlines() if line.strip()), "")
                    writer.writerow([res.id, res.gold, r.rank, r.ref_id, r.figure_index, int(r.from_gold_doc),
                                     int(r.correct), " ".join(map(str, r.numbers)), first[:200]])

        print(f"\n=== prompt: {name} ===")
        print(f"{'id':<6}{'gold':>9}  reads (rank: chart -> numbers; * = right number, g = gold document)")
        for res in results:
            cells = [f"{r.rank}: {r.ref_id}/fig{r.figure_index:02d} -> {','.join(map(str, r.numbers)) or '-'}"
                     f"{'*' if r.correct else ''}{'g' if r.from_gold_doc else ''}" for r in res.reads]
            print(f"{res.id:<6}{res.gold:>9g}  " + "  |  ".join(cells))
        print(f"per-question detail -> {out}")

    print(f"\nstrategy={args.strategy}{' (oracle doc)' if args.oracle_doc else ''}  "
          f"questions={summaries[args.prompt[0]]['questions']}  reads/question={n}  tolerance={args.tolerance:g}")
    print(f"{'':<20}" + "".join(f"{name:>10}" for name in args.prompt))
    for key in metrics:
        print(f"  {key:<18}" + "".join(f"{summaries[name][key]:>10.3f}" for name in args.prompt))
    print(f"  {'read errors':<18}" + "".join(f"{summaries[name]['read_errors']:>10}" for name in args.prompt))
    save_summary(EVAL_OUT / f"summary__e2e__{args.strategy}{'__oracle' if args.oracle_doc else ''}.json", "e2e",
                 {"strategy": args.strategy, "oracle_doc": args.oracle_doc, "reads": n, "tolerance": args.tolerance,
                  "model": model, "dense": not args.no_dense, "prompts": summaries})


if __name__ == "__main__":
    main()
