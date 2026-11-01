"""Write the pull request description from measured results, so no number in it is typed by hand.

`build_body` is a pure function of the facts gathered by scripts/make_pr_body.py (eval summaries saved by the
eval scripts, chunk and reading counts, the git diff, the test run). Anything that is missing, produced at a
different commit than the one being proposed, or produced from uncommitted changes is flagged in the text and
returned as a warning; nothing is filled in with a guess.
"""
from __future__ import annotations

SHARED_PREFIXES = ("src/wattbot/chunking/",)
SHARED_FILES = ("src/wattbot/snapshot.py", "src/wattbot/config.py", "src/wattbot/embeddings.py",
                "src/wattbot/vectorstore.py", "pyproject.toml", "uv.lock", ".gitignore")

COMMANDS = {
    "retrieval": "uv run python scripts/eval_retrieval.py --retriever hybrid --strategies hybrid hybrid_fig",
    "figures": "uv run python scripts/eval_figures.py --retriever hybrid",
    "e2e": "uv run python scripts/eval_figures_e2e.py --reads 3 --prompt default scope",
    "e2e_oracle": "uv run python scripts/eval_figures_e2e.py --reads 3 --oracle-doc",
}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def delta(new, old) -> str:
    if new is None or old is None:
        return "n/a"
    return f"{new - old:+.3f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


class Body:
    """Accumulates markdown and the warnings that go with it."""

    def __init__(self, head: str):
        self.head, self.parts, self.warnings = head, [], []

    def add(self, text: str = "") -> None:
        self.parts.append(text)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def provenance(self, name: str, summary: dict) -> str:
        """'(commit abc1234, 2026-11-02)' plus a visible flag if the result doesn't belong to this commit."""
        text = f"commit `{summary.get('git_commit', '?')}`, {summary.get('created_at', '?')[:10]}"
        problems = []
        if summary.get("git_commit") != self.head:
            problems.append(f"produced at `{summary.get('git_commit', '?')}` but the PR is at `{self.head}`")
        if summary.get("git_dirty"):
            problems.append("produced with uncommitted changes")
        for problem in problems:
            self.warn(f"{name}: {problem}; rerun it")
        return f"_({text})_" + ("  **⚠ " + "; ".join(problems) + "; rerun before merging.**" if problems else "")

    def missing(self, name: str, command: str) -> None:
        self.warn(f"{name}: no result found")
        self.add(f"**⚠ No result for {name}.** Run `{command}`, then regenerate this description.")


def _retrieval_section(b: Body, summary: dict | None) -> None:
    b.add("### Text questions: does adding figure chunks hurt? (`eval_retrieval.py`)")
    if not summary:
        return b.missing("the text-question regression check", COMMANDS["retrieval"])
    strategies = summary["strategies"]
    if not {"hybrid", "hybrid_fig"} <= set(strategies):
        return b.missing("the text-question regression check (needs both `hybrid` and `hybrid_fig`)",
                         COMMANDS["retrieval"])
    cols = ["doc_recall@5", "passage_recall@1", "passage_recall@5", "passage_recall@10", "passage_recall@20",
            "passage_mrr"]
    old, new = strategies["hybrid"], strategies["hybrid_fig"]
    b.add(f"`{summary['retriever']}`, {summary['questions']} train questions "
          f"({summary['passage_questions']} with verbatim evidence located). "
          + b.provenance("text-question check", summary))
    b.add()
    b.add(table(["strategy", "chunks"] + cols,
                [["`hybrid`", fmt(old["chunks"])] + [fmt(old[c]) for c in cols],
                 ["`hybrid_fig`", fmt(new["chunks"])] + [fmt(new[c]) for c in cols],
                 ["change", f"{int(new['chunks']) - int(old['chunks']):+d}"] + [delta(new[c], old[c]) for c in cols]]))
    b.add()
    b.add("Figure chunks compete with text chunks in the same ranking, so a small change here is the cost of "
          "that competition; the text chunks themselves are unchanged.")


def _figures_section(b: Body, summary: dict | None) -> None:
    b.add("### Chart questions: retrieval (`eval_figures.py`)")
    if not summary:
        return b.missing("the chart-question retrieval check", COMMANDS["figures"])
    strategies = summary["strategies"]
    first = next(iter(strategies.values()))
    cols = ["figure_recall@1", "figure_recall@5", "figure_recall@10", "value_recall@1", "value_recall@5",
            "value_recall@10"]
    b.add(f"`{summary['retriever']}`; {first['figure_questions']} questions name a figure, "
          f"{first['value_questions']} have a numeric answer. " + b.provenance("chart retrieval check", summary))
    b.add()
    b.add(table(["strategy", "figure chunks"] + cols,
                [[f"`{name}`", fmt(s["figure_chunks"])] + [fmt(s[c]) for c in cols] for name, s in strategies.items()]))
    b.add()
    b.add("`figure_recall` = the figure the question's supporting text names is in the top k; `value_recall` = a "
          "top-k chunk of the right document contains the numeric answer (a number repeated elsewhere also counts).")


def _e2e_section(b: Body, normal: dict | None, oracle: dict | None) -> None:
    b.add("### Chart questions: end to end (`eval_figures_e2e.py`)")
    if not normal:
        return b.missing("the end-to-end check", COMMANDS["e2e"])
    n = normal["reads"]
    columns = {name: s for name, s in normal["prompts"].items()}
    if oracle:
        columns.update({f"{name}, oracle doc": s for name, s in oracle["prompts"].items()})
    else:
        b.warn("end-to-end oracle-document run: no result found")
    questions = next(iter(normal["prompts"].values()))["questions"]
    b.add(f"{questions} numeric chart questions, `{normal['model']}`, top {n} charts read per question, number must "
          f"equal the gold answer at its precision" + (f" (tolerance {normal['tolerance']:g})" if normal["tolerance"]
                                                       else "") + ". " + b.provenance("end-to-end check", normal))
    b.add()
    metrics = ["gold_doc_read@1", f"gold_doc_read@{n}", "answer@1", f"answer@{n}", "strict_answer@1",
               f"strict_answer@{n}", "wrong_number@1", "read_errors"]
    b.add(table(["metric"] + list(columns),
                [[f"`{m}`"] + [fmt(s.get(m)) for s in columns.values()] for m in metrics]))
    b.add()
    if not oracle:
        b.add(f"_Oracle-document run missing: `{COMMANDS['e2e_oracle']}`._")
        b.add()
    first = next(iter(normal["prompts"].values()))
    found_first = round(first["gold_doc_read@1"] * questions) if first.get("gold_doc_read@1") is not None else None
    if found_first is not None:
        b.add(f"- The right paper is the first chart's source for {found_first}/{questions} questions, and "
              f"{round(first[f'gold_doc_read@{n}'] * questions)}/{questions} within the top {n}.")
    b.add("- `answer@k` = one of the first k charts read gave the right number; `strict_answer@k` also requires that "
          "chart to be from a gold document; `wrong_number@1` = the first chart gave a confident wrong number.")
    if len(normal["prompts"]) > 1:
        b.add("- Prompts were compared on the same retrieved charts. Keep the default unless the table shows the other "
              "one is better on `wrong_number@1` without losing `answer@k`.")


def build_body(facts: dict) -> tuple[str, list[str]]:
    """Return (markdown, warnings) for the PR description."""
    b = Body(facts["head"])
    readings, chunks = facts.get("readings"), facts.get("chunks")

    b.add("## Summary")
    b.add("- Charts were indexed by caption only. This adds a vision-model pass that reads each chart (type, axes, "
          "legend, data table, summary) and indexes it as an extra `figure` chunk, and a way to re-read the chart "
          "image to answer a question.")
    if readings:
        b.add(f"- {readings['count']} charts were read with `{readings['model']}`: {readings['charts']} indexed as "
              f"figure chunks, {readings['dropped']} logos/photos dropped"
              + (f", {readings['invalid']} replies were not valid JSON" if readings["invalid"] else "") + ".")
    else:
        b.warn("no chart readings found in documents/figure_text")
    if chunks and chunks["identical"] is True:
        b.add(f"- Existing strategies are unchanged. Figure chunks are additive: the {chunks['text']} text chunks of "
              f"`hybrid_fig` are identical to `hybrid`'s, plus {chunks['figure']} figure chunks.")
    elif chunks and chunks["identical"] is False:
        b.add(f"- **⚠ The text chunks of `hybrid_fig` differ from `hybrid`'s** ({chunks['text']} text chunks, "
              f"{chunks['figure']} figure chunks).")
        b.warn("text chunks of hybrid_fig are not identical to hybrid's")
    elif chunks:
        b.add(f"- {chunks['figure']} figure chunks added (text-chunk identity with `hybrid` not checked: "
              "`documents/chunks/hybrid.jsonl` missing).")
    else:
        b.warn("documents/chunks/hybrid_fig.jsonl not found")
    b.add("- Two opt-in strategies: `hybrid_fig` (vision-model readings) and `hybrid_figlabels` (labels from the PDF "
          "text layer, no model, as a baseline).")
    b.add("- Chart readings are a snapshot part (`figure_text`), so teammates can skip the vision run. "
          "Design, run steps and limits: `docs/figure_retrieval_design.md`.")
    b.add()

    b.add("## Shared files touched")
    files = facts.get("files") or []
    shared = [f for f in files if f in SHARED_FILES or f.startswith(SHARED_PREFIXES)]
    b.add(", ".join(f"`{f}`" for f in shared) if shared else "None.")
    untouched = [f for f in ("pyproject.toml", "uv.lock") if f not in files]
    if untouched:
        b.add(" and ".join(f"`{f}`" for f in untouched) + " " + ("are" if len(untouched) > 1 else "is") + " unchanged.")
    b.add()

    b.add("## Results")
    b.add()
    _retrieval_section(b, facts.get("retrieval"))
    b.add()
    _figures_section(b, facts.get("figures"))
    b.add()
    _e2e_section(b, facts.get("e2e"), facts.get("e2e_oracle"))
    b.add()

    b.add("## Known limits and follow-ups")
    figures = facts.get("figures")
    if figures:
        first = next(iter(figures["strategies"].values()))
        b.add(f"- Small samples: {first['figure_questions']} figure-naming and {first['value_questions']} numeric "
              "chart questions, so one question moves a rate by 7-10 points.")
    b.add("- Choosing the right chart inside the right paper is the weak step. Next experiments: match on caption and "
          "summary only, then a cross-encoder rerank of the top figure chunks.")
    b.add("- The gateway's embeddings of one text differ slightly between calls, so older scripts' rankings can move "
          "between runs; `ask_figures.py` and `eval_figures_e2e.py` cache query vectors to be repeatable.")
    b.add()

    b.add("## Test plan")
    tests = facts.get("tests")
    if tests and tests["failed"] == 0 and tests["passed"] > 0:
        b.add(f"- [x] `pytest tests/`: {tests['passed']} passed (offline, fake gateway client)")
    elif tests:
        b.add(f"- [ ] **⚠ `pytest tests/`: {tests['passed']} passed, {tests['failed']} failed**")
        b.warn("tests did not all pass")
    else:
        b.add("- [ ] `pytest tests/` (not run by the generator)")
        b.warn("tests were not run")
    if chunks and chunks["identical"] is not None:
        b.add(f"- [{'x' if chunks['identical'] else ' '}] `hybrid_fig` text chunks identical to `hybrid`'s")
    b.add("- [ ] `scripts/check_vision_model.py` passes against the gateway")
    b.add("- [ ] Readings spot-checked against their images with `scripts/review_figures.py`")
    b.add("- [ ] One approval from someone other than the author (CONTRIBUTING.md)")

    return "\n".join(b.parts).rstrip() + "\n", b.warnings
