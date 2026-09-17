# Contributing

We're 7 people, each running a coding agent, on one Kaggle challenge. That means more
branches, more commits, and bigger diffs than usual — these conventions exist to keep
`main` clean and reviewable despite that volume. Keep them short and actually follow them.

## Branches

`<name>/<short-topic>`, e.g. `evan/pdf-parser-retry`, `mei/feature-lag-windows`.
One branch per topic, not per person for the whole marathon — delete branches after merge.

## Pull requests

- **One concern per PR.** A bug fix, one feature, or one experiment — not "various improvements."
- **Target under ~400 changed lines**, excluding `uv.lock` and data/notebook diffs. If your
  agent produced more, that's a sign to split the PR before opening it, not after review comments.
- Include what changed and why, and how you verified it (test run, script output, a plot).
  If an agent wrote most of the diff, say so and note what you checked before opening it.
- Squash-merge to `main` so history stays one commit per PR.

## Review

- Every PR needs **one human approval from someone other than the author** before merging —
  agent-generated code still needs a second set of eyes, especially on anything touching
  shared modules, feature engineering, or the scoring/submission pipeline.
- The author is responsible for reading their own agent's diff before requesting review, not
  just before submitting the agent's output.
- Small PRs get reviewed fast; that's the main reason to keep them small.

## What an agent can touch without asking first

Free to do without a heads-up in chat:
- Files under `src/wattbot/` you're actively working in, `scripts/`, your own notebook,
  and tests.
- Adding a new file in the right place per the structure below.

Ask (a message in the team channel or PR description is enough) before an agent:
- Edits `pyproject.toml`, `.gitignore`, CI config, or anything else shared repo-wide.
- Touches another person's in-progress branch or notebook.
- Regenerates `uv.lock` for a reason other than a dependency it just added (run `uv lock`,
  don't hand-edit it).
- Deletes or renames files it didn't create.

## Avoiding collisions

- Before starting, post in the team channel which file(s)/area you're working on.
- Keep PRs small and merge often — the longer a branch lives, the more likely someone
  else touched the same file.
- Notebooks are single-owner: prefix with your name (`notebooks/evan_eda.ipynb`) and
  don't edit someone else's. Shared logic that two people need belongs in `src/wattbot/`,
  not a notebook.
- Rebase onto `main` before opening a PR so conflicts surface on your branch, not in review.

## Commit messages

`<type>: <short summary>`, e.g. `feat: add lag-window features to training set`,
`fix: handle missing PDF pages in parser`. Types: `feat`, `fix`, `data`, `refactor`, `docs`,
`chore`. Body (optional) explains *why*, not what — the diff already shows what. No need to
mention that an agent wrote it; the author is accountable for what they commit either way.

## Repo structure

```
src/wattbot/     installable package: data loading, features, models, shared utils
scripts/         one-off/CLI entry points (download_papers.py, parse_pdfs.py, ...)
notebooks/       exploration, one file per person, prefixed with your name
tests/           tests for src/wattbot/
docs/            write-ups, EDA notes, model cards
data/            gitignored — raw/intermediate data, never committed
submissions/     gitignored — generated Kaggle submission CSVs
```

New code that's reusable goes in `src/wattbot/`, not copy-pasted across notebooks or scripts.
If you're not sure where something goes, ask before scattering it at repo root.
