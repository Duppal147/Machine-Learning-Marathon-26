# Machine-Learning-Marathon-26
RAGGEDY AMP

## BadgerBrain gateway setup

The embedding and chat models are served by the UW-Madison BadgerBrain gateway
(`https://llm-gw01.doit.wisc.edu/v1`), which requires the GlobalProtect VPN. Settings are read from
environment variables in `src/wattbot/config.py`:

```bash
export OPENAI_API_KEY=...                     # your BadgerBrain key
export EMBEDDING_MODEL=qwen3-vl-embedding-8b  # default
python scripts/check_text_embeddings.py       # live check: dimension, stability, similarity ordering
```

## Documents to searchable chunks

```bash
python scripts/download_papers.py WattBot2026/metadata.csv --out-dir documents/pdfs \
    --metadata-out WattBot2026/metadata_downloaded.csv  # file -> ref_id mapping used by chunking
python scripts/parse_pdfs.py <doc_id> [...]             # PDF -> documents/parsed_json (Docling)
python scripts/chunk_docs.py --strategy hybrid          # -> documents/chunks/hybrid.jsonl
python scripts/vector_index.py index --strategy hybrid  # embed + store in documents/chroma (incremental)
python scripts/vector_index.py search "How much water did US data centers consume in 2023?"
```

Step-by-step instructions (which scripts to run, in what order): [docs/pipeline_runbook.md](docs/pipeline_runbook.md).
Design notes and retrieval results: [docs/chunking_design.md](docs/chunking_design.md).

## Sharing the processed corpus (skip parsing and embedding)

`documents/` is gitignored, so the parsed JSON, chunks and Chroma index are shared as snapshots
published as GitHub Releases (`index-<timestamp>`) on this repo:

```bash
python scripts/fetch_index.py --list      # available snapshots
python scripts/fetch_index.py             # install the latest into documents/ (no GitHub login needed)
python scripts/fetch_index.py --force     # replace existing local data (kept as *.bak-<timestamp>)

python scripts/publish_index.py           # build documents/snapshots/wattbot-index-<ts>.tar.gz
python scripts/publish_index.py --upload  # ... and publish it (needs `gh auth login` + write access)
```

Snapshots include full document text and this repo is public, so anyone can download them.
Everyone should use the chromadb version pinned in `uv.lock`, since the index files are version-specific.
