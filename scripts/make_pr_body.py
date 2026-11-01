#!/usr/bin/env python3
"""Write documents/pr_body.md for the chart-retrieval PR from measured results; nothing in it is typed by hand.

Usage:
    uv run --with pytest python scripts/make_pr_body.py [--run] [--base origin/main] [--out documents/pr_body.md]
                                                        [--skip-tests] [--strict]

Reads the summaries the eval scripts save in documents/eval/ (summary__*.json, each stamped with the git
commit that produced it), counts the chart readings and figure chunks, checks that the text chunks of
`hybrid_fig` are identical to `hybrid`'s, lists the shared files in the diff against --base, and runs the
tests. A result that is missing, from another commit, or from uncommitted changes is flagged in the text
and listed at the end; --strict exits non-zero if there is any such warning.

--run first runs the steps that produce the results: chunk_docs, vector_index, eval_retrieval, eval_figures
and eval_figures_e2e (with and without --oracle-doc). They need the gateway (VPN + key) and take a few minutes
plus about 30 vision calls on the first run. Commit your changes first, so the results are stamped with the
commit being proposed. Pass --with pytest to uv so the tests can run.

Then: gh pr create --base main --head <branch> --title "..." --body-file documents/pr_body.md
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import figures
from wattbot.chunking import ChunkStore
from wattbot.eval.summary import git_state, load_summary
from wattbot.pr_body import build_body

EVAL = Path("documents/eval")
CHUNKS = Path("documents/chunks")

RUN_STEPS = [
    ["scripts/chunk_docs.py", "--strategy", "hybrid_fig"],
    ["scripts/vector_index.py", "index", "--strategy", "hybrid_fig", "--prune"],
    ["scripts/eval_retrieval.py", "--retriever", "hybrid", "--strategies", "hybrid", "hybrid_fig"],
    ["scripts/eval_figures.py", "--retriever", "hybrid"],
    ["scripts/eval_figures_e2e.py", "--reads", "3", "--prompt", "default", "scope"],
    ["scripts/eval_figures_e2e.py", "--reads", "3", "--oracle-doc"],
]


def readings_facts() -> dict | None:
    records = []
    for path in sorted(figures.FIGURE_TEXT_DIR.glob("*/figure_*.json")):
        record = figures.load_record(path)
        if record and "result" in record:
            records.append(record)
    if not records:
        return None
    kinds = [r["result"].get("kind", "unknown") for r in records]
    return {"count": len(records),
            "model": Counter(r.get("model", "?") for r in records).most_common(1)[0][0],
            "charts": sum(k not in figures.NON_CONTENT_KINDS for k in kinds),
            "dropped": sum(k in figures.NON_CONTENT_KINDS for k in kinds),
            "invalid": sum(bool(r["result"].get("parse_error")) for r in records)}


def chunk_facts() -> dict | None:
    fig_path, base_path = CHUNKS / "hybrid_fig.jsonl", CHUNKS / "hybrid.jsonl"
    if not fig_path.exists():
        return None
    chunks = list(ChunkStore.load(fig_path))
    text = [(c.text, c.embed_text) for c in chunks if c.kind != "figure"]
    identical = None
    if base_path.exists():
        identical = text == [(c.text, c.embed_text) for c in ChunkStore.load(base_path)]
    return {"figure": sum(c.kind == "figure" for c in chunks), "text": len(text), "identical": identical}


def changed_files(base: str) -> list[str]:
    for ref in (base, "main"):
        result = subprocess.run(["git", "diff", "--name-only", f"{ref}...HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            return [line for line in result.stdout.splitlines() if line]
    return []


def run_tests() -> dict | None:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], capture_output=True, text=True)
    if "No module named pytest" in result.stderr + result.stdout:
        print("pytest isn't available here; run through `uv run --with pytest` to include the test count",
              file=sys.stderr)
        return None
    count = lambda word: int(next(iter(re.findall(rf"(\d+) {word}", result.stdout)), 0))
    return {"passed": count("passed"), "failed": count("failed") + count("error")}


def newest_summary(pattern: str, needs=(), prefer: str = "chroma") -> dict | None:
    """The most recent saved summary matching the glob, preferring a retriever whose name contains `prefer`."""
    found = [s for s in (load_summary(p) for p in EVAL.glob(pattern)) if s]
    found = [s for s in found if all(n in s.get("strategies", {}) for n in needs)]
    found.sort(key=lambda s: (prefer in s.get("retriever", ""), s.get("created_at", "")), reverse=True)
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="run chunking, indexing and the evals first")
    parser.add_argument("--base", default="origin/main", help="branch the PR targets")
    parser.add_argument("--out", type=Path, default=Path("documents/pr_body.md"),
                        help="default is in the gitignored documents/ folder, so it can't be committed by accident")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 if anything is missing or stale")
    args = parser.parse_args()

    if args.run:
        for step in RUN_STEPS:
            print(f"\n$ python {' '.join(step)}", flush=True)
            if subprocess.run([sys.executable, *step]).returncode != 0:
                sys.exit(f"error: `{' '.join(step)}` failed; fix it and rerun")

    head, dirty = git_state()
    facts = {
        "head": head,
        "readings": readings_facts(),
        "chunks": chunk_facts(),
        "files": changed_files(args.base),
        "tests": None if args.skip_tests else run_tests(),
        "retrieval": newest_summary("summary__retrieval__*.json", needs=("hybrid", "hybrid_fig")),
        "figures": newest_summary("summary__figures__*.json"),
        "e2e": load_summary(EVAL / "summary__e2e__hybrid_fig.json"),
        "e2e_oracle": load_summary(EVAL / "summary__e2e__hybrid_fig__oracle.json"),
    }
    body, warnings = build_body(facts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(body)

    print(f"wrote {args.out} (PR at commit {head}{', with uncommitted changes' if dirty else ''})")
    if dirty:
        warnings.append("uncommitted changes in the working tree: results may not match what the PR will contain")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    if not warnings:
        print("  no warnings: every number comes from a result produced at this commit")
    if warnings and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
