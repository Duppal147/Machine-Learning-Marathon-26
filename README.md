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

Design notes and retrieval results are in [docs/chunking_design.md](docs/chunking_design.md).
