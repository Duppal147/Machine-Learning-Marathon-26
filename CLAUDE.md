# CLAUDE.md — rules for coding agents on this repo

Read this before touching anything. `CONTRIBUTING.md` covers team process (branches, PR size,
review); this file covers what an agent must and must not do, and the competition facts that
make a change correct or wrong. Where the two overlap, `CONTRIBUTING.md` wins on process.

## Never

- **Never commit or push to `main`.** Work on `<name>/<topic>`, open a PR, get one human
  approval. No exceptions, including one-line fixes and "just adding a file".
- **Never force-push, rebase, or amend a branch you did not create.** Merge `main` in instead.
- **Never hardcode an answer to a question, or read from `train_QA.csv` at inference time.**
  Top-3 teams get re-run at a pinned tag and checked for answer leakage and hardcoding; a
  submission that does not reproduce is removed from the standings. This is the one rule that
  can disqualify the whole team.
- **Never edit `data/metadata.csv`, `data/train_QA.csv`, or `data/test_Q.csv`.** They are
  competition inputs and read-only. Derived files go elsewhere.
- **Never commit** `.env`, API keys, downloaded PDFs, `chroma_db/`, or submission CSVs.
- **Never tune on the test set.** `test_Q.csv` is touched to produce a submission, nothing else.
- **Never delete, skip, or weaken a test to make a run pass.** Fix the cause or say it's broken.
- **Never introduce unseeded randomness or a non-zero temperature** in the submission path —
  see Determinism below.

## Ask a human first

- Changing `pyproject.toml`, `requirements.txt`, `uv.lock`, `.gitignore`, or CI.
- Changing prompts, model IDs, chunk size, or retrieval parameters — these move the score
  silently, and nobody can tell from the diff whether it helped.
- Deleting or renaming a file you did not create.
- Anything touching another person's in-progress branch.

## Green light

Code in the module you're working in, its tests, docstrings, and new files placed per the
structure in `CONTRIBUTING.md`.

## The competition contract

These come from the rules and from `train_QA.csv` (245 rows, the authoritative example).
Getting any of them wrong costs score regardless of how good retrieval is.

**Score** = `0.75 × answer_value` + `0.20 × citation F1` + `0.05 × is_NA handling`.

- `answer_value` and `ref_id` are scored **independently**. A wrong number still earns citation
  credit. So on an answerable question, always cite — never emit `is_blank` because the model
  was unsure of the value.
- **Abstain only when the corpus genuinely cannot answer.** Only 5 of 245 training rows are
  `is_NA`. Defaulting to `is_blank` on a parse failure or a low retrieval score throws away
  real questions.
- **`is_blank` is all-or-nothing.** On an abstention row, `answer_value`, `ref_id`, `ref_url`
  and `supporting_materials` must *all* be `is_blank`. `explanation` must stay non-empty or
  submission validation fails.
- **Numerics are scored at ±0.1% relative tolerance.** Do not round anywhere in the pipeline.
  Carry full precision and let the scorer's tolerance absorb it.
- **Two bracket conventions, opposite meanings.** `[low,high]` in gold is *their* tolerance
  band around a derived answer — submit a single number. `(low,high)` is a real range answer —
  submit both endpoints. Never emit a bracketed range against a band; it scores 0.
- **True/False answers are `1` / `0`** in `answer_value`, not the words.
- **Gold `ref_id` looks like `['luccioni2025c']`**, sometimes multi-doc
  (`['patterson2021', 'jegham2025']`). Match the training file's format exactly.
- **Cite exactly the supporting documents — no more, no fewer.** F1 punishes padding. Retrieve
  widely; cite narrowly. Citing only documents actually quoted in `supporting_materials` is the
  default rule.
- **Evidence-type flags in `train_QA.csv`** (`Quote` 194, `Table` 38, `Math` 25, `CrossPaper` 19,
  `Figure` 17, `Reconcile` 11, `is_NA` 5) — stratify any dev slice across these. A random 30
  questions tells you nothing about figure or math performance.

## Scoring authority

`Score.py` from the Kaggle Data tab is the only scorer that counts, and **it is not in this
repo yet** — download it before trusting any number. `src/wattbot/scoring.py` is an unofficial
reimplementation for quick sanity checks; it can agree with itself and still disagree with the
leaderboard. Never report a score from it without saying which scorer produced it.

## Determinism

The submission path must reproduce exactly from a clean checkout:

- `temperature=0` on every model call in the submission path. Query expansion currently runs at
  `0.7` — that makes runs unreproducible and needs fixing before any submission is pinned.
- Seed anything random. No wall-clock or ordering dependence.
- Caches are an optimization, never a dependency — the pipeline must produce the same
  predictions cold.
- Record for every run: git SHA, config, per-question predictions, and the score.

## Commands

```bash
pip install -r requirements.txt        # authoritative dep list today, not pyproject.toml
cp .env.example .env                   # BadgerBrain key; requires GlobalProtect VPN

python scripts/run_pipeline.py ingest --limit 3
python scripts/run_pipeline.py run --questions data/train_QA.csv --output train_pred.csv --limit 5
python scripts/run_pipeline.py score --gold data/train_QA.csv --pred train_pred.csv
pytest tests/ -v                       # offline, no API needed
```

`requirements.txt` and `pyproject.toml` currently disagree and `pyproject.toml` has no
build-system section, so `uv sync` produces an environment that cannot import `wattbot`. Use
`requirements.txt` until that is fixed; don't "fix" it silently in an unrelated PR.

## Where code goes

Per `CONTRIBUTING.md`: reusable logic in `src/wattbot/`, entry points in `scripts/`, tests
mirroring the source layout in `tests/`. Before adding a downloader, parser, or scorer, check
whether one already exists — the repo has picked up duplicates of all three, and adding a
fourth is worse than fixing the split.
