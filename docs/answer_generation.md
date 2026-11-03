# Generate and evaluate WattBot answers

This is an initial text-evidence answer baseline. It reuses the existing chunks, Qwen
vectors, BM25/RRF fusion and cross-encoder reranker. It adds a chat call and validated
Kaggle CSV export; it never reads gold answers into the generation prompt.

## Setup

Run from the repo root with Python >=3.14 and the shared snapshot installed. On the
current Mac, keep the older embedding environment intact:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/documents/venv-query-eval"
uv sync --locked
```

Keep the UW GlobalProtect VPN connected for gateway access. Inject credentials for
each command using your existing 1Password setup. Other teammates may use their own
credential injection; the code reads `OPENAI_API_KEY` only from its process environment.

List available models, then select a **chat/instruct** model (not the embedding model):

```bash
OPENAI_API_KEY="op://<vault>/<item>/<field>" \
op run -- uv run python scripts/generate_answers.py --list-models

export CHAT_MODEL="<available-chat-model>"
```

The default retrieval is equal-weight RRF(BM25, Qwen), k_rrf=60, depth=50, followed
by `BAAI/bge-reranker-large` over 20 candidates. Five clean passages go to generation.
Use `uv run --with sentence-transformers` for the reranker; its first use downloads
the model (about 2.1 GB). No project dependency/lockfile change is required.
`--retriever bm25 --no-rerank` is a cheaper diagnostic baseline with different retrieval.

The document index must already match the chunks and embedding model. The runner
refuses to re-embed missing/stale document vectors; fetch a matching snapshot or index
explicitly first. Generation does not change chunking or the embedding implementation.

## Start with ten training questions

```bash
OPENAI_API_KEY="op://<vault>/<item>/<field>" \
op run -- uv run --with sentence-transformers python scripts/generate_answers.py \
  --questions WattBot2026/train_QA.csv \
  --run-dir documents/runs/train-smoke-v1 \
  --limit 10 --snapshot-tag index-20261007-113647
```

The loader uses **only `id`, `question`, `answer_unit`**. It retains unanswerable rows.
The model returns JSON containing answer, value, explanation and quoted evidence
identified by chunk ID. Source IDs and URLs are derived from chunk metadata, not
trusted from the model. Quotes must occur literally in the cited passage, allowing
whitespace differences. This checks provenance, not the semantic correctness of an answer.
Abstentions must use `is_blank` and empty evidence; the expected unit is preserved.

Each run saves:

- `run.json`: model, prompt/code/lockfile hashes, Git SHA, input questions, corpus hash,
  retrieval settings, optional snapshot tag; **no credential**.
- `answers/*.json`: completed answers, raw model responses and retrieved passages.
- `errors.json`: invalid responses or technical failures for inspection.
- `status.json`: completed, failed and pending counts.
- `predictions.csv`: written only after every selected question has a valid checkpoint
  and output IDs match the selected inputs exactly. A smoke run is only the selected subset.

Progress is printed per question. Transient chat errors are retried with bounded backoff.
Malformed answers remain failures; they are never converted to abstentions. Technical
failures stop further requests, leaving later questions pending.

Re-run the **same command with `--resume`** after interruption or failure. Completed
answers are validated and reused; failed/pending questions are retried. Different inputs,
models, prompts, corpus or code require a new run directory. A `.lock` prevents concurrent
writers. After a hard kill, remove that lock only after confirming the old process stopped.
The final `predictions.csv` appears only on success (exit code 0).
The initial output budget is 8,192 tokens, including reasoning when the chat model
counts it. By default a `length` response is retried once with twice that budget,
using the same model, prompt and evidence. `--length-retries 0` disables recovery;
`--length-retries 2` allows two doublings. Attempt budgets and finish reasons are saved
in each checkpoint. Exhausted retries remain errors, never fake abstentions. An
incomplete response records its finish reason and raw content in `errors.json`.
Changing token budgets or retry policy requires a fresh run directory.

## Score locally, then prepare the test set

```bash
uv run python WattBot2026/Score.py \
  WattBot2026/train_QA.csv documents/runs/train-smoke-v1/predictions.csv
```

The official scorer joins matching IDs: the command above scores **only ten smoke
questions**, not the full training set or Kaggle test. Inspect failures before scaling.
Reserve a fixed, representative training holdout before prompt development. Prepare
separate development/holdout question files and gold files; generation may receive only
the question/expected-unit input, while `Score.py` reads the corresponding gold file.
Do not optimize prompts on test questions or select settings using Kaggle test feedback.

After fixing settings and validating on the holdout, use a fresh run directory for
all test questions (omit `--limit`):

```bash
OPENAI_API_KEY="op://<vault>/<item>/<field>" \
op run -- uv run --with sentence-transformers python scripts/generate_answers.py \
  --questions WattBot2026/test_Q.csv --run-dir documents/runs/test-v1 \
  --snapshot-tag index-20261007-113647
```

Review `status.json` and the complete CSV before manually uploading to Kaggle. This
runner never submits anything. Test answers are hidden; local scoring uses labeled
training data only. Text-only evidence may miss figures and parsing errors.

## Recorded baseline (2026-10-29)

| Evaluation | Scope | Score |
|---|---|---:|
| Official local scorer | Fixed 50-question training holdout | 0.8759 |
| Kaggle public leaderboard | Public subset of all 317 submitted test questions | 0.823 |

The holdout had 43/50 correct answer values and citation F1 0.9047. Its two
unanswerable questions were correctly refused. These are different evaluation
sets; the score difference is not a measured improvement or regression.

The test submission uses `qwen3.8-27b`, temperature 0, the default hybrid retrieval
and query instruction, 20 reranking candidates and 5 supplied passages. Snapshot
`index-20261007-113647` provides 6,033 hybrid chunks and matching Qwen vectors.
The initial output allowance is 8,192 tokens. Seven questions needed 16,384;
one needed 32,768. Only output-length recovery changed, with the same model,
prompt and evidence. Use `--length-retries 2` for this allowance policy:

```bash
OPENAI_API_KEY="op://<vault>/<item>/<field>" \
op run -- uv run --with sentence-transformers python scripts/generate_answers.py \
  --questions WattBot2026/test_Q.csv --run-dir documents/runs/test-baseline \
  --chat-model qwen3.8-27b --max-output-tokens 8192 --length-retries 2 \
  --snapshot-tag index-20261007-113647
```

All 317 predictions passed ID, question/unit preservation, quotation and citation
metadata checks; 43 were abstentions. Kaggle submission reference: `57778407`.
The submitted CSV SHA-256 is
`27e6501b331f78ada3132da933f4c29520700dff23564c6a849466000c807889`.
Private test labels remain hidden. Checkpoints, splits, CSVs and receipts stay in
the ignored local `documents/runs/` directory; no downloaded competition data or
generated outputs are included in the source change.

## Offline verification

```bash
uv run --with pytest pytest tests/
python scripts/generate_answers.py --help
```

All generation tests mock model calls. They cover gold-field exclusion, quote/source
validation, abstention, malformed/truncated responses, complete ID coverage, checkpoint
integrity, failures, resume without repeated model calls, and an offline BM25-to-CSV run.
