#!/usr/bin/env python3
"""Generate evidence-grounded Kaggle answers; see docs/answer_generation.md.

Examples (credentials must already be injected, e.g. with op run):
    python scripts/generate_answers.py --list-models
    python scripts/generate_answers.py --questions WattBot2026/train_QA.csv \
        --run-dir documents/runs/smoke --chat-model MODEL --limit 10
    # Re-run the same command with --resume after an interruption or recorded failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot import config
from wattbot.chunking import ChunkStore
from wattbot.embeddings import create_embedding_client
from wattbot.generation import (COLUMNS, SYSTEM_PROMPT, GenerationValidationError, generate_answer, load_input_questions,
                                passage_context, validate_answer, validate_predictions)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_write(path, write):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def save_json(path, value):
    atomic_write(path, lambda stream: json.dump(value, stream, indent=2, ensure_ascii=False))


@contextmanager
def run_lock(folder):
    lock = folder / ".lock"
    try:
        stream = lock.open("x")
    except FileExistsError:
        raise ValueError(f"run is locked: {lock}; ensure no process is using it before removing this lock") from None
    with stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def build_retriever(args, chunks, corpus_hash):
    from wattbot.eval import BM25Retriever, CachedRetriever, ChromaRetriever, CrossEncoderReranker, FusionRetriever
    if args.retriever == "bm25":
        retriever = BM25Retriever()
    else:
        from wattbot.vectorstore import ChunkVectorStore, embed_hash
        store = ChunkVectorStore(args.strategy)
        existing = store.collection.get(include=["metadatas"])
        hashes = {key: (meta or {}).get("embed_hash")
                  for key, meta in zip(existing["ids"], existing["metadatas"])}
        if any(hashes.get(c.chunk_id) != embed_hash(c, store.model) for c in chunks):
            raise ValueError("document index is missing or stale; fetch a matching snapshot or index it explicitly first")
        cache_key = fingerprint({"corpus": corpus_hash, "model": store.model, "base_url": config.BASE_URL,
                                 "instruction": None if args.no_query_instruction else config.QUERY_INSTRUCTION})
        dense = CachedRetriever(ChromaRetriever(args.strategy, query_instruction=not args.no_query_instruction,
                                                store=store),
                                Path("documents/eval/cache") / f"generation__{cache_key}.json")
        retriever = FusionRetriever([BM25Retriever(), dense]) if args.retriever == "hybrid" else dense
    if not args.no_rerank:
        # Separate this cache by corpus so changed chunk text cannot reuse old pair scores.
        retriever = CrossEncoderReranker(retriever, args.reranker, candidates=args.candidates,
                                         cache_dir=Path("documents/eval/rerank_cache") / corpus_hash)
    return retriever


def run_questions(questions, chunks, retriever, generate, run_dir, settings, *, resume=False,
                  generation_trace=None):
    """Checkpoint validated answers atomically. Retry failed IDs on resume, never fill fake rows."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run.json"
    signature = fingerprint(settings)
    if manifest_path.exists():
        if not resume:
            raise ValueError("run already exists; use --resume or choose a new --run-dir")
        if json.loads(manifest_path.read_text())["fingerprint"] != signature:
            raise ValueError("run settings, inputs or code changed; choose a new --run-dir")
    elif any(run_dir.iterdir()):
        raise ValueError("run directory is not empty and has no manifest")
    with run_lock(run_dir):
        if not manifest_path.exists():
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            save_json(manifest_path, {"fingerprint": signature, "settings": settings,
                                      "git_sha": revision.stdout.strip(),
                                      "created_at": datetime.now(timezone.utc).isoformat()})
        answers = run_dir / "answers"
        answers.mkdir(exist_ok=True)
        by_id = {c.chunk_id: c for c in chunks}
        rows, errors, fitted = [], {}, False
        for position, question in enumerate(questions, 1):
            checkpoint = answers / f"{fingerprint(question['id'])}.json"
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                if record["question"] != question:
                    raise ValueError("checkpoint question differs from the current input")
                context = [by_id[key] for key in record["passage_ids"]]
                row = validate_answer(question, record["raw_response"], context)
                if row != record["row"]:
                    raise ValueError("checkpoint row differs from its validated response")
                rows.append(row)
                print(f"[{position}/{len(questions)}] {question['id']}: cached", flush=True)
                continue
            started = time.monotonic()
            context = []
            if generation_trace is not None:
                generation_trace.clear()
            try:
                if not fitted:
                    retriever.fit(chunks)
                    fitted = True
                ranked = retriever.search([question["question"]], settings["top_k"])
                if len(ranked) != 1:
                    raise ValueError("retriever returned the wrong number of query results")
                context = [chunks[index] for index in ranked[0]]
                row, raw = generate(question, context)
                # Validate even injected generators before a checkpoint becomes trusted.
                if row != validate_answer(question, raw, context):
                    raise ValueError("generator row differs from its validated response")
                save_json(checkpoint, {"question": question, "row": row, "raw_response": raw,
                                       "passage_ids": [c.chunk_id for c in context],
                                       "passages": passage_context(context),
                                       "generation": dict(generation_trace or {}),
                                       "seconds": round(time.monotonic() - started, 3)})
                rows.append(row)
                print(f"[{position}/{len(questions)}] {question['id']}: saved ({time.monotonic()-started:.1f}s)",
                      flush=True)
            except Exception as error:
                message = str(error)
                if config.API_KEY:
                    message = message.replace(config.API_KEY, "[REDACTED]")
                raw = getattr(error, "raw_response", None)
                if raw and config.API_KEY:
                    raw = raw.replace(config.API_KEY, "[REDACTED]")
                errors[question["id"]] = {"type": type(error).__name__, "message": message,
                                           "raw_response": raw, "passages": passage_context(context),
                                           "generation": dict(generation_trace or {})}
                save_json(run_dir / "errors.json", errors)
                print(f"[{position}/{len(questions)}] {question['id']}: FAILED ({type(error).__name__})", flush=True)
                if not isinstance(error, GenerationValidationError):
                    # Connectivity/authentication/setup failures usually affect the whole run.
                    # Leave later questions pending instead of repeating a gateway outage.
                    break
        save_json(run_dir / "errors.json", errors)
        save_json(run_dir / "status.json", {"requested": len(questions), "completed": len(rows),
                                             "failed": len(errors),
                                             "pending": len(questions) - len(rows) - len(errors),
                                             "complete": not errors})
        if errors:
            print(f"Saved {len(rows)}/{len(questions)} answers; {len(errors)} failed. "
                  "Review errors.json and resume.", flush=True)
            return 1
        validate_predictions(questions, rows)
        def write_csv(stream):
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        atomic_write(run_dir / "predictions.csv", write_csv)
        print(f"Complete: {len(rows)} predictions -> {run_dir / 'predictions.csv'}", flush=True)
        return 0


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--chat-model", default=config.CHAT_MODEL)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--chunks", type=Path, help="default: documents/chunks/<strategy>.jsonl")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--retriever", choices=("bm25", "chroma", "hybrid"), default="hybrid")
    parser.add_argument("--reranker", default="BAAI/bge-reranker-large")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--candidates", type=positive, default=20)
    parser.add_argument("--top-k", type=positive, default=5)
    parser.add_argument("--max-output-tokens", type=positive, default=8192)
    parser.add_argument("--length-retries", type=int, choices=(0, 1, 2), default=1,
                        help="retry truncated responses with twice the token budget (default: once)")
    parser.add_argument("--no-query-instruction", action="store_true")
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot-tag", help="optional source snapshot tag, recorded for reproducibility")
    args = parser.parse_args(argv)
    if args.list_models:
        client = create_embedding_client().with_options(timeout=20, max_retries=0)
        for model in sorted(item.id for item in client.models.list().data):
            print(model)
        return 0
    if not args.questions or not args.run_dir or not args.chat_model.strip():
        parser.error("--questions, --run-dir and --chat-model (or CHAT_MODEL) are required")
    if args.chat_model == config.EMBEDDING_MODEL:
        parser.error("select a chat/instruct model, not the embedding model")
    if not args.no_rerank and args.top_k > args.candidates:
        parser.error("--candidates must be at least --top-k")
    questions = load_input_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    chunks_path = args.chunks or Path("documents/chunks") / f"{args.strategy}.jsonl"
    chunks = list(ChunkStore.load(chunks_path))
    if not chunks:
        raise ValueError("chunk corpus is empty")
    corpus_hash = checksum(chunks_path)
    settings = {"questions": questions, "corpus_sha256": corpus_hash, "snapshot_tag": args.snapshot_tag,
                "chat_model": args.chat_model, "base_url": config.BASE_URL, "temperature": 0,
                "max_output_tokens": args.max_output_tokens, "length_retries": args.length_retries,
                "retriever": args.retriever,
                "strategy": args.strategy, "embedding_model": config.EMBEDDING_MODEL,
                "query_instruction": None if args.no_query_instruction else config.QUERY_INSTRUCTION,
                "chroma_dir": str(Path(config.CHROMA_DIR).resolve()),
                "reranker": None if args.no_rerank else args.reranker, "candidates": args.candidates,
                "top_k": args.top_k, "prompt_sha256": fingerprint(SYSTEM_PROMPT),
                "generator_sha256": checksum(Path(__file__).resolve().parents[1] / "src/wattbot/generation.py"),
                "runner_sha256": checksum(__file__),
                "lock_sha256": checksum(Path(__file__).resolve().parents[1] / "uv.lock")}
    retriever = build_retriever(args, chunks, corpus_hash)
    client = None
    generation_trace = {}
    def generate(question, passages):
        nonlocal client
        generation_trace.clear()
        if client is None:
            client = create_embedding_client().with_options(max_retries=0)
        return generate_answer(question, passages, model=args.chat_model, client=client,
                               max_tokens=args.max_output_tokens, length_retries=args.length_retries,
                               trace=generation_trace)
    return run_questions(questions, chunks, retriever, generate, args.run_dir, settings, resume=args.resume,
                         generation_trace=generation_trace)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit("Interrupted; completed checkpoints are saved. Re-run with --resume.")
    except Exception as error:
        message = str(error).replace(config.API_KEY, "[REDACTED]") if config.API_KEY else str(error)
        sys.exit(f"error: {type(error).__name__}: {message}")
