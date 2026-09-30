# Pipeline Runbook: Which Scripts to Run, in What Order

This takes you from nothing to a searchable index of the WattBot corpus. Run every command from the **repo root**. For how and why the pieces work, see [chunking_design.md](chunking_design.md).

```
                           ┌─────────────── shortcut: 0b. fetch_index.py ───────────────┐
                           │                                                            ▼
download_papers.py ──► parse_pdfs.py ──► chunk_docs.py ──► vector_index.py index ──► vector_index.py search
 (PDFs + ref_id map)    (Docling JSON)    (chunks JSONL)     (embeddings → Chroma)     eval_retrieval.py
                                                                     │
                                                                     └──► publish_index.py (share with team)
```

| Step | Script | Reads | Writes | Needs VPN + key? | Time (full corpus) |
|---|---|---|---|---|---|
| 1 | `download_papers.py` | `WattBot2026/metadata.csv` | `documents/pdfs/`, `WattBot2026/metadata_downloaded.csv` | no | minutes |
| 2 | `parse_pdfs.py` | `documents/pdfs/` | `documents/parsed_json/`, `documents/figures/` | no | **hours** |
| 3 | `chunk_docs.py` | `documents/parsed_json/` | `documents/chunks/<strategy>.jsonl` | no | seconds |
| 4 | `check_text_embeddings.py` | nothing | nothing (live gateway check) | **yes** | ~1–2 min |
| 5 | `vector_index.py index` | `documents/chunks/` | `documents/chroma/` | **yes** | ~1,000+ API calls |
| 6 | `vector_index.py search` / `eval_retrieval.py` | chunks, Chroma, `WattBot2026/train_QA.csv` | `documents/eval/` | chroma: **yes**; tfidf: no | minutes |
| 7 | `publish_index.py` | all of the above | a GitHub Release | no (needs `gh`) | seconds |

Everything under `documents/` and `WattBot2026/` is gitignored and stays on your machine. Step 7 is how the team shares it.

---

## Step 0: One-time setup

### 0a. Environment and data
```bash
uv sync                                   # installs docling, chromadb, openai, scikit-learn, ...
```
- **Competition files.** Download the WattBot 2026 competition data from Kaggle into `WattBot2026/`. The steps below need `WattBot2026/metadata.csv` and `WattBot2026/train_QA.csv`.
- **Gateway access (steps 4–6).** Connect to the UW GlobalProtect VPN and export your BadgerBrain key. `src/wattbot/config.py` reads these variables:
  ```bash
  export OPENAI_API_KEY=...                        # BadgerBrain key
  # optional overrides:
  # export EMBEDDING_MODEL=qwen3-vl-embedding-8b
  # export CHROMA_DIR=documents/chroma
  ```

### 0b. Shortcut: start from a teammate's snapshot
If someone has already published a snapshot, fetch it and skip straight to **step 6**. The snapshot contains the parsed JSON, the chunks, the Chroma index (if it was built) and the `ref_id` metadata:
```bash
uv run python scripts/fetch_index.py --list      # see what's available
uv run python scripts/fetch_index.py             # install the latest into documents/ and WattBot2026/
```
It refuses to overwrite local data. Add `--force` to replace it; your old data is moved to `*.bak-<timestamp>`. You only need steps 1–5 when you're adding documents or changing how chunks are made.

---

## Step 1: Download the PDFs
```bash
uv run python scripts/download_papers.py WattBot2026/metadata.csv \
    --out-dir documents/pdfs \
    --metadata-out WattBot2026/metadata_downloaded.csv
```
- Always pass both flags. The defaults (`pdfs/`, `metadata.csv`) don't match where the later steps look.
- `metadata_downloaded.csv` maps each PDF file name to its WattBot `ref_id` (e.g. `amazon2023`). Chunking uses it to attach citations.
- Files that already exist are skipped, so re-running only fetches what's missing.

## Step 2: Parse PDFs with Docling
The script takes document IDs, which are the PDF file names without `.pdf`:
```bash
uv run python scripts/parse_pdfs.py 2109.04459 2404.07413
```
Parse **only the PDFs that haven't been parsed yet**. The script re-parses anything you pass it:
```bash
todo=$(for f in documents/pdfs/*.pdf; do id=$(basename "$f" .pdf); \
       [ -f "documents/parsed_json/$id.json" ] || echo "$id"; done)
uv run python scripts/parse_pdfs.py $todo
```
- This is the slow step: accurate table recognition takes minutes per long report.
- Run it in batches or in the background, e.g. `nohup uv run python scripts/parse_pdfs.py $todo > parse.log 2>&1 &`.
- It writes `documents/parsed_json/<id>.json` plus figure PNGs in `documents/figures/<id>/`.

## Step 3: Chunk the parsed documents
```bash
uv run python scripts/chunk_docs.py --strategy hybrid          # all parsed docs -> documents/chunks/hybrid.jsonl
uv run python scripts/chunk_docs.py --strategy hybrid 2109.04459   # only some docs
```
- **Strategies:** `hybrid` (default, recommended), `hybrid_raw`, `section`, `fixed`. `--max-tokens N` changes the chunk budget.
- **Output:** per-document chunk counts and sizes are printed. Chunk IDs look like `2109.04459::hybrid::0005`.
- **Re-chunking:** this is cheap, so re-run it whenever parsed documents are added or the chunking code changes.

## Step 4: Check the embedding gateway (VPN + key)
```bash
uv run python scripts/check_text_embeddings.py
```
It should end with `overall: PASS`. The first call after a cold start can take ~90 s.

## Step 5: Embed chunks into ChromaDB (VPN + key)
```bash
uv run python scripts/vector_index.py index --strategy hybrid
```
- **Output:** the chunks are stored in `documents/chroma/`, in the collection `hybrid__qwen3-vl-embedding-8b`.
- **Incremental:** only new or changed chunks are embedded. Run it twice and the second run reports `embedded=0`.
- **Interrupted runs:** re-run the same command. It resumes and doesn't redo finished batches.
- `--prune` also deletes stored chunks that are no longer in the JSONL. Use it after re-chunking.
- `--rebuild` drops the collection and re-embeds everything, e.g. after changing the embedding model or query setup.

## Step 6: Use and evaluate the index
Search it:
```bash
uv run python scripts/vector_index.py search "What was the total consumptive water use in 2023?" -k 5
uv run python scripts/vector_index.py search "..." --ref-id shehabi2024        # one document only
```

Measure retrieval quality on the training questions:
```bash
uv run python scripts/eval_retrieval.py                                        # TF-IDF, all strategies, offline
uv run python scripts/eval_retrieval.py --retriever chroma --strategies hybrid hybrid_raw fixed
uv run python scripts/eval_retrieval.py --retriever chroma --strategies hybrid --no-query-instruction
```
- **What's scored:** only questions whose source documents have been parsed. The header shows how many.
- **Per-question results:** written to `documents/eval/<strategy>__<retriever>.csv`.
- **Side effect:** `--retriever chroma` indexes any missing chunks first (step 5), using the chunks it rebuilds from `documents/parsed_json/`.

## Step 7: Share your results with the team
```bash
uv run python scripts/publish_index.py            # build documents/snapshots/wattbot-index-<ts>.tar.gz (local only)
uv run python scripts/publish_index.py --upload   # ... and publish it as GitHub Release index-<ts>
```
- `--upload` needs `gh auth login` and write access to the team repo.
- Publish after step 5 so the Chroma index is included. An empty index is skipped automatically.
- **The team repo is public**, so anyone can download a published snapshot, including the full document text.

---

## Common workflows

| I want to… | Run |
|---|---|
| Get started quickly | 0a → `fetch_index.py` → step 6 |
| Add newly downloaded papers | step 1 → step 2 (new IDs only) → step 3 → step 5 → (step 7) |
| Change chunking code or settings | step 3 → `vector_index.py index --prune` → step 6 (compare with `eval_retrieval.py`) |
| Try a different embedding model | `export EMBEDDING_MODEL=...` → step 4 → step 5. It uses a new collection, so the old one is kept. |
| Run the tests | `uv run --with pytest pytest tests/` (not bare `pytest`, which also collects stray scripts at the repo root) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: OPENAI_API_KEY must be set` | Export your BadgerBrain key (step 0a). |
| Timeouts or connection errors from the gateway | Check the VPN. The first call can take ~90 s. Batches are retried 3× automatically, and re-running resumes. |
| Chunks have no `ref_id` in their metadata | Step 1 wasn't run with `--metadata-out WattBot2026/metadata_downloaded.csv`, or a new PDF isn't in that CSV. Re-run step 1, then step 3. |
| `eval_retrieval.py` scores only a few questions | Only parsed documents count. Parse more (step 2). |
| `fetch_index.py`: `local data would be replaced` | Add `--force`. Existing data is kept as `*.bak-<timestamp>`. |
| Chroma errors after someone upgraded dependencies | Everyone should use the chromadb version in `uv.lock` (`uv sync`), since index files are version-specific. |
