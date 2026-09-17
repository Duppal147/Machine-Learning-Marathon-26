# WattBot 2026 — ML Marathon 26

RAG pipeline for the [WattBot 2026 Kaggle competition](https://www.kaggle.com/competitions/WattBot2026).
Answers 500+ questions about AI's environmental impact from a corpus of 122 papers and reports.

## Pipeline stages

```
metadata.csv (122 docs)
    │
    ▼
┌─────────┐   ┌──────────────┐   ┌────────────────┐
│ Download │──▶│ Extract text │──▶│ Chunk (400w,   │
│ PDFs    │   │ (PyMuPDF)    │   │ 80w overlap)   │
└─────────┘   └──────────────┘   └───────┬────────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │ Embed chunks   │
                                 │ (BadgerBrain)  │
                                 └───────┬───────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │ ChromaDB store │
                                 └───────┬───────┘
                                         │
    Question ──▶ Query expansion ──▶ Embed queries
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │ Vector search  │──▶ Re-rank ──▶ Top-K chunks
                                 └───────────────┘
                                                            │
                                                            ▼
                                                    ┌──────────────┐
                                                    │ Augment prompt│──▶ Generate answer
                                                    └──────────────┘         │
                                                                             ▼
                                                                     ┌──────────────┐
                                                                     │ Postprocess   │
                                                                     │ (normalize,   │
                                                                     │  validate)    │
                                                                     └──────┬───────┘
                                                                            │
                                                                            ▼
                                                                     submission.csv
```

## Setup

**Requirements:** Python 3.10+, UW-Madison GlobalProtect VPN (for BadgerBrain API access).

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your BadgerBrain API key
```

## Usage

### 1. Ingest corpus (requires VPN)
```bash
python scripts/run_pipeline.py ingest
python scripts/run_pipeline.py ingest --limit 3   # test with 3 docs
```

### 2. Run pipeline on questions (requires VPN)
```bash
python scripts/run_pipeline.py run --questions data/train_QA.csv --output train_pred.csv --limit 5
python scripts/run_pipeline.py run --questions data/test_Q.csv --output submission.csv
```

### 3. Score locally
```bash
python scripts/run_pipeline.py score --gold data/train_QA.csv --pred train_pred.csv
```

## Scoring

WattBot Score = 0.75 * answer_accuracy + 0.20 * citation_F1 + 0.05 * abstention_accuracy

The built-in scorer (`src/wattbot/scoring.py`) is for quick sanity checks.
For leaderboard-accurate scoring, download `Score.py` from the Kaggle
competition Data tab and run:
```bash
python Score.py data/train_QA.csv my_predictions.csv
```

## Tests

```bash
pytest tests/ -v
```

## Project structure

```
data/               Competition data (metadata, train, test CSVs)
src/wattbot/
  config.py         BadgerBrain endpoint + model config (from env vars)
  ingest.py         PDF download -> text extraction -> chunking -> embedding -> ChromaDB
  retriever.py      Query expansion -> embedding -> vector search -> re-rank
  generator.py      Prompt augmentation -> LLM generation -> JSON parsing
  postprocess.py    Answer normalization, ref_id cleaning, is_blank enforcement
  pipeline.py       End-to-end orchestrator
  scoring.py        Local WattBot Score implementation
scripts/
  run_pipeline.py   CLI entrypoint
tests/              Unit tests (run offline, no API needed)
```

## BadgerBrain connection

- Gateway: `https://llm-gw01.doit.wisc.edu/v1`
- Chat model: `qwen3.8-27b`
- Embedding model: `qwen3-vl-embedding-8b` (4096-dim)
- Vision model: `churro-3b` (not yet wired in)
- Requires: GlobalProtect VPN + NetID firewall allowlisting
- First call may take ~90s (cold start); timeout is set to 300s

## What's not built yet

- **Vision pipeline** — `churro-3b` for Figure-type questions (config wired, not implemented)
- **Answer validation** — second LLM pass to verify answer against evidence
- **End-to-end evaluation** — needs VPN + full ingestion run first
