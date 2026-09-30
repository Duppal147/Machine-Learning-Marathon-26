# Chunking Pipeline: Design Notes

Status: implemented (`src/wattbot/chunking/`, `src/wattbot/eval/`), first iteration · Last updated: 2026-09-30

This document explains how parsed documents become retrievable chunks for the WattBot RAG system: how the pipeline works, why it is built this way, how to run and extend it, and what to improve next.

---

## 1. Goals

1. Turn parsed documents (currently Docling JSON in `documents/parsed_json/`) into chunks with **stable IDs** and **citation metadata** (pages, section path, source item refs).
2. Keep the **input format** swappable. Structured markdown or another parser should need only a new loader.
3. Keep the **chunking strategy** and **recall tricks** (such as prefixing section headings) swappable and comparable side by side.
4. Keep it simple: plain dataclasses and small Protocol interfaces, with no framework dependency.

## 2. What the input data looks like

These observations about the Docling output drive most of the design:

| Observation | Consequence |
|---|---|
| Every `section_header` is `level: 1`. Docling does not recover heading nesting from PDFs. | The heading hierarchy is rebuilt from the numbering in the heading text (§4.2). |
| `content_layer: "furniture"` marks running page headers and footers (652 of 3,363 items in the Amazon report). | Furniture is dropped. |
| `texts` parented under a `picture` are labels read out of the chart image (903 across 5 docs). | Only picture **captions** are kept. |
| Tables are cell grids in `tables[].data.grid` with caption refs. | Tables are rendered to markdown with the caption prepended, and kept whole. |
| Lists live in `groups[]` of `list_item`s. | Groups are walked recursively, and list items get a `- ` prefix. |
| Every item has `prov[0].page_no` and a `self_ref` (`#/texts/42`). | These are kept as `pages` and `source_refs` for citations and debugging. |
| Some `section_header`s are noise: infographic stats ("680M", "#1"), captions ("Table 3.1 …"), reference fragments (". Public Law …"). | Heading sanity checks demote these to text or caption blocks. |

## 3. Architecture

```
            ┌──────────────┐   Document    ┌──────────────┐           ┌──────────┐         ┌──────────────┐           ┌───────────┐
 file(s) ──►│   Loader     │──(Blocks)───► │ Block filters│──Blocks──►│ Chunker  │─Chunks─►│ Chunk filters│──Chunks──►│ Enrichers │──► ChunkStore (.jsonl)
            │ (per format) │               │ DropSections │           │ strategy │         │  MinTokens   │           │ heading   │
            └──────────────┘               └──────────────┘           └──────────┘         └──────────────┘           │ prefix …  │
                                                                                                                         └───────────┘
```

Each stage is a small `Protocol`, so any object with the right method can be dropped in (strategy pattern). `ChunkingPipeline` wires the stages together and assigns IDs. `REGISTRY` names preset combinations so they can be picked from the CLI.

### Key design decisions

1. **One intermediate format (`Block`) between loaders and chunkers.** Loaders are the only code that knows about Docling. Chunkers, filters and enrichers see only `Block`/`Chunk`. A new input format means one new loader class.
2. **Chunking and enrichment are separate.** A `Chunk` carries:
   - `text`: the clean passage shown to the LLM and quoted in answers. Enrichers never change it.
   - `embed_text`: what gets embedded. Recall enrichers change only this field; metadata enrichers change only `metadata`.

   This allows A/B tests of recall tricks (heading prefix, LLM context, …) on identical chunk boundaries. `hybrid` and `hybrid_raw` are exactly that pair.
3. **Chunks never cross a section boundary** (except in the deliberately section-blind `fixed` baseline). This keeps each chunk about one topic and makes `heading_path` accurate.
4. **Tables are atomic.** Splitting a table separates its values from their headers, so a table stays whole even when it is over the token budget.
5. **Filters run at two points.** `DropSections` runs on blocks, before chunking, so even the section-blind chunker never sees the bibliography. `MinTokens` runs on the finished chunks.
6. **The pipeline assigns IDs after filtering.** This keeps them dense and deterministic.

## 4. Components

All code lives in `src/wattbot/chunking/`. CLIs live in `scripts/`.

### 4.1 Data model (`models.py`)

| Type | Fields | Notes |
|---|---|---|
| `Block` | `doc_id, kind, text, heading_path, page, source_refs` | `kind` ∈ `text, list_item, table, caption, formula, code`. `heading_path` excludes the document title. |
| `Document` | `doc_id, title, blocks, metadata` | `metadata` holds `source_path` and `format`. |
| `Chunk` | `chunk_id, doc_id, text, embed_text, heading_path, pages, kind, source_refs, strategy, metadata` | `kind` ∈ `text, table`. `metadata` holds `title`, the document metadata, and the WattBot fields `ref_id, url, year, doc_type, ref_title` (§4.5). |

### 4.2 Loaders (`loaders.py`)

`DocumentLoader.load(path) -> Document`

**`DoclingJsonLoader`** walks `body.children` in reading order and resolves `$ref`s into `texts / groups / tables / pictures`.

**Heading inference (`HeadingTracker`).** It keeps a stack of `(level, heading)` and sets the level from the heading text:

| Pattern | Example | Level |
|---|---|---|
| Roman numeral, but only if it continues the sequence (so `C.` after `II.` is a letter) | `III. Results` | 1 |
| Capital letter | `A. Model Sparsification` | 2 under a roman section, else 1 |
| Dotted number (1–2 digits per part, so "2023 Year in Review" is not numbering) | `3.1.2 Detail` | number of parts |
| Appendix, or a name in `TOP_LEVEL_NAMES` (Abstract, References, Acknowledgments, Contents, …) | `References` | 1 (resets nesting) |
| Anything else | `Server Types` | one below the last numbered heading |

**Title detection.** The title is the first unnumbered, non-"Abstract" heading on page 1, before any section starts. The fallback is the Docling `name`.

**Headings that are rejected** (their text is still kept, just not treated as a heading):
- fewer than 3 letters
- starts with a non-alphanumeric character
- starts with `Table` / `Figure` / `Fig.`, which becomes a caption instead

### 4.3 Chunkers (`chunkers.py`)

`Chunker.chunk(blocks) -> list[Chunk]`. Length is measured by a pluggable `len_fn`. The default `approx_tokens` is words × 1.3.

| Chunker | Behaviour |
|---|---|
| `HybridChunker(max_tokens=400, min_tokens=50, tables_separate=True)` | Groups blocks by `heading_path` and fills chunks up to `max_tokens`. Paragraphs over budget are split on sentence boundaries, with a word-window fallback. Tables become their own chunk. A final chunk shorter than `min_tokens` is merged into the one before it if the pair fits in `max_tokens + min_tokens`. |
| `SectionChunker(max_tokens=1200)` | The same code with tables inline and no tail merging, so in practice one chunk per section, split only if the section is very long. |
| `FixedSizeChunker(max_tokens=300, overlap=50)` | A sliding word window over the whole document that ignores sections. This is the baseline. `heading_path` is taken from the first block in the window. |

### 4.4 Filters (`filters.py`)

- `DropSections(names=DEFAULT_DROPPED_SECTIONS)` (block filter) drops everything under References, Bibliography, Citations and Bibliography, Contents / Table of Contents, List of Figures and List of Tables. Section numbering is ignored when matching.
- `MinTokens(min_tokens=5)` (chunk filter) drops fragments.

### 4.5 Enrichers (`enrichers.py`)

`Enricher.enrich(chunk, doc) -> Chunk`. An enricher changes `embed_text` or `metadata`, never `text`.

- `HeadingPrefixEnricher(include_title=True, sep=" > ")` produces:
  ```
  SONIC: A Sparse Neural Network … > III. Software and Dataflow Optimizations > A. Model Sparsification

  To generate SpNNs, we adapt a layer-wise, sparsity-aware training …
  ```
- `CorpusMetadataEnricher(csv_path="WattBot2026/metadata_downloaded.csv")` joins the chunk's `doc_id` (file stem) to the CSV's `local_path` stem. It adds `ref_id` (the id WattBot answers cite, e.g. `amazon2023`), `url`, `year`, `doc_type` and `ref_title` to `chunk.metadata`. If the CSV is missing or the document is unknown, the chunk is left unchanged. Every preset runs it first.

### 4.6 Pipeline and registry (`pipeline.py`)

- `AutoLoader` picks a loader by file extension (`{".json": DoclingJsonLoader()}`).
- `ChunkingPipeline(name, chunker, loader, enrichers, block_filters, chunk_filters)`, with `.run(paths)` and `.run_document(doc)`.
- `REGISTRY` presets:

| Name | Chunker | Heading prefix | Default `max_tokens` | Purpose |
|---|---|---|---|---|
| `hybrid` | Hybrid | yes | 400 | Recommended default |
| `hybrid_raw` | Hybrid | no | 400 | Isolates the effect of the heading prefix |
| `section` | Section | yes | 1200 | Larger, topic-complete chunks |
| `fixed` | FixedSize | no | 300 | Section-blind baseline |

All presets use `DropSections`, `MinTokens` and `CorpusMetadataEnricher`.

### 4.7 Store (`store.py`)

`ChunkStore` saves chunks to and loads them from JSONL (one chunk per line). It provides:
- `get(chunk_id)`
- `by_doc(doc_id)`
- `neighbors(chunk_id, k)`: the chunk plus `k` on each side in document order, for widening the context at answer time.

### 4.8 Chunk IDs

`{doc_id}::{strategy}::{index:04d}`, e.g. `2109.04459::hybrid::0005`.

- The same inputs and preset always give the same IDs.
- `doc_id` is the parsed file's stem (the arXiv id or `slug_hash`), **not** the WattBot `ref_id`. Use `chunk.metadata["ref_id"]` for citations in answers.
- Changing `--max-tokens` keeps the preset name, so IDs and the output file are reused. Add a new registry entry to keep variants side by side.

## 5. How to run

### Command line

```bash
# default: hybrid, all docs in documents/parsed_json -> documents/chunks/hybrid.jsonl
uv run python scripts/chunk_docs.py

# pick a preset and budget
uv run python scripts/chunk_docs.py --strategy section --max-tokens 800

# only some documents (file stems in documents/parsed_json)
uv run python scripts/chunk_docs.py 2109.04459 2404.07413

# generate every preset for comparison
for s in hybrid hybrid_raw section fixed; do uv run python scripts/chunk_docs.py --strategy $s; done
```

The CLI prints each document's chunk count and median/max token length, then writes `documents/chunks/<strategy>.jsonl`.

Full pipeline from scratch:

```bash
uv run python scripts/parse_pdfs.py 2109.04459 2404.07413   # PDF -> documents/parsed_json + documents/figures
uv run python scripts/chunk_docs.py                         # JSON -> documents/chunks/hybrid.jsonl
```

### From Python

```python
# run from the repo root with src on the path, e.g. PYTHONPATH=src uv run python
from wattbot.chunking import ChunkStore, build_pipeline

# build chunks in memory
chunks = build_pipeline("hybrid", max_tokens=300).run(["documents/parsed_json/2109.04459.json"])

# or load a saved run and look chunks up by id
store = ChunkStore.load("documents/chunks/hybrid.jsonl")
c = store.get("2109.04459::hybrid::0005")
c.text, c.embed_text, c.heading_path, c.pages, c.metadata["title"]
store.neighbors(c.chunk_id, k=1)   # surrounding context
```

A custom pipeline without touching the registry:

```python
from wattbot.chunking import ChunkingPipeline, HybridChunker, HeadingPrefixEnricher, DropSections, MinTokens

pipe = ChunkingPipeline(
    "hybrid_250_notitle",
    HybridChunker(max_tokens=250, min_tokens=30),
    enrichers=[HeadingPrefixEnricher(include_title=False)],
    block_filters=[DropSections()],
    chunk_filters=[MinTokens(10)],
)
```

### Embedding and storing chunks (ChromaDB)

`src/wattbot/vectorstore.py` embeds each chunk's `embed_text` with `wattbot.embeddings.embed_texts` (the Qwen gateway model), and stores the vector in a persistent ChromaDB collection in `documents/chroma/`. The chunk `text` and flat citation metadata are stored alongside it.

**Collections.** There is one collection per strategy and model, e.g. `hybrid__qwen3-vl-embedding-8b`, so different chunkings or models never mix. Cosine space; Chroma's own embedding function and telemetry are both turned off.

**Stored metadata.** Chroma only accepts scalar values, so lists are flattened:
- `doc_id`, `ref_id`, `title`, `url`, `year`, `doc_type`, `kind`, `strategy`
- `heading_path`, joined with `" > "`
- `pages` (`"3,4"`) and `page_start` (int)
- `embed_hash`

**Incremental and resumable.** `embed_hash` is sha256(model + embed_text), and only new or changed chunks are embedded. Each batch of 64 is written as soon as it is embedded, so an interrupted run resumes where it stopped. Transient gateway errors (timeouts, dropped connections, 429/5xx) are retried 3 times with backoff.

**Asymmetric queries.** Following the Qwen3 embedding convention, queries are embedded as `Instruct: <config.QUERY_INSTRUCTION>\nQuery: <question>` and documents are embedded as-is. Pass `instruction=None` (CLI `--no-query-instruction`) to compare.

The chunks JSONL stays the source of truth. Search hits carry `chunk_id`, so `ChunkStore.get()` and `neighbors()` still work.

```bash
# needs OPENAI_API_KEY + UW VPN (see README)
python scripts/vector_index.py index --strategy hybrid            # incremental; re-runs embed only changes
python scripts/vector_index.py index --strategy hybrid --prune    # also delete chunks no longer in the JSONL
python scripts/vector_index.py index --strategy hybrid --rebuild  # drop the collection and re-embed all
python scripts/vector_index.py search "What was the total consumptive water use in 2023?" -k 5
python scripts/vector_index.py search "..." --ref-id shehabi2024  # restrict to one document
```

```python
from wattbot.vectorstore import ChunkVectorStore
store = ChunkVectorStore("hybrid")
hits = store.search("How much energy did training JetMoE use?", k=5, where={"doc_type": "paper"})
hits[0].chunk_id, hits[0].score, hits[0].metadata["ref_id"], hits[0].metadata["pages"]
```

Rough size for the full corpus: ~15–20k chunks × 4096 float32 ≈ 250–330 MB, and ~1,000–1,300 gateway calls for a first full index.

### Evaluating retrieval (comparing strategies)

```bash
# all presets, offline TF-IDF retriever (a few seconds)
uv run python scripts/eval_retrieval.py

# a subset, with a different budget
uv run python scripts/eval_retrieval.py --strategies hybrid fixed --max-tokens 250

# dense retrieval through an OpenAI-compatible embeddings endpoint (e.g. the UW gateway)
export OPENAI_API_KEY=$(op read op://Credentials/MLM26-RaggedyAmp_elinck/credential)
uv run python scripts/eval_retrieval.py --retriever embed --embed-model <model-id> --base-url https://llm-gw01.doit.wisc.edu/v1
```

`src/wattbot/eval/retrieval.py` rebuilds each preset's chunks from `documents/parsed_json`, retrieves the top chunks for each question in `WattBot2026/train_QA.csv`, and reports:

| Metric | Meaning |
|---|---|
| `doc_recall@k` | a top-k chunk comes from a gold `ref_id` document |
| `passage_recall@k` | a top-k chunk from a gold doc contains the verbatim evidence |
| `passage_mrr` | mean reciprocal rank of the first evidence-bearing chunk (0 if not in the top 20) |
| `ctx_tokens@5` | mean tokens in the top 5 chunks, i.e. the context budget that recall costs |

How it works:
- **Question scope.** Only questions citing at least one parsed document are scored (`is_NA` questions are skipped). Coverage grows automatically as more PDFs are parsed.
- **Evidence extraction** (`extract_evidence`) pulls verbatim segments out of `supporting_materials`. It prefers double-quoted spans. Otherwise it uses the whole text, split at `refid:` markers. `...`, `…` and ` -- ` are treated as elisions. Segments shorter than 6 words are dropped.
- **Evidence matching** (`evidence_coverage`) is the fraction of a segment's words found in order in the chunk (difflib on word lists). It ignores case, punctuation and Docling's doubled spaces. A chunk "contains" the evidence at coverage ≥ 0.7.
- **Passage metrics** only count questions whose evidence is found in *some* chunk of a gold doc. Evidence that is paraphrased, figure-only, or from a document that isn't parsed yet can't be located, so it is excluded rather than counted as a miss.

Per-question results (doc rank, passage rank, top-5 chunk IDs) go to `documents/eval/<strategy>__<retriever>.csv` for failure analysis.

The embedding retriever has not yet been run against the gateway: the 1Password CLI needs an interactive unlock. Confirm the embedding model id there first.

### Tests

```bash
uv run --with pytest pytest tests/test_chunking.py tests/test_eval.py
```

- **Chunking tests:** heading inference, the loader (furniture, picture text, tables, lists, references), the Hybrid chunker's rules, the fixed-window overlap, IDs and a store round trip, plus spot checks on `2109.04459`.
- **Evaluation tests:** evidence extraction and matching, the metadata join, and scoring on a toy corpus.

Tests that need `documents/` or `WattBot2026/` are skipped if the data is missing.

## 6. How to extend

| To add… | Do this |
|---|---|
| A new input format (e.g. markdown) | Write a class with `load(path) -> Document` that emits `Block`s with `heading_path` filled in (markdown `#` depth gives levels directly), then register its extension in `AutoLoader` in `pipeline.py`. |
| A chunking strategy | Write a class with `chunk(blocks) -> list[Chunk]` (reuse `blocks_to_chunk` / `split_text`) and add a `REGISTRY` entry. |
| A recall trick | Write an `Enricher` that returns `dataclasses.replace(chunk, embed_text=...)`, and add a preset that uses it. |
| A content filter | A `keep(block)` or `keep(chunk)` object, added to `block_filters` / `chunk_filters`. |
| An exact token budget | Pass `len_fn=lambda s: len(tokenizer.encode(s))` to the chunker. |

## 7. Future improvements

Roughly in priority order.

### 7.1 ~~Map `doc_id` to the WattBot `ref_id`~~ (done)
Implemented as `CorpusMetadataEnricher` (§4.5). Still open: consider putting `ref_id` in the chunk ID.

### 7.2 ~~Retrieval evaluation harness~~ (done, with follow-ups)
Implemented as `scripts/eval_retrieval.py` / `src/wattbot/eval/` (§5). Follow-ups:
- Run it with the gateway's embedding model; TF-IDF only approximates what dense retrieval rewards.
- Parse more PDFs. Only 33 of 245 answerable questions are covered by the 5 parsed documents, so one question is worth about 3.5 points of recall.
- Break results down by the `Table` / `Figure` / `Math` / `CrossPaper` flags (already loaded into `Question.flags`).
- Cache embeddings by `(model, embed_text hash)` so repeated runs don't re-embed unchanged chunks.
- Score on `test_Q.csv` answers once an end-to-end answer step exists (`WattBot2026/Score.py`).

### 7.3 Contextual retrieval enricher
An `LLMContextEnricher` would ask the chat model (the Qwen gateway, see `test_qwen.py`) for one or two sentences placing the chunk in the document, and prepend them to `embed_text`. This matters most for chunks with weak headings (the Amazon report). Cache results on disk, keyed by chunk text hash, to avoid repeat LLM calls.

### 7.4 Better heading hierarchy
- Use layout signals: bbox height/font size from `prov`, or Docling's own hierarchy post-processing if it becomes available, to nest unnumbered headings (the Amazon report is currently flat).
- Handle `Appendix A` / `A.1` numbering explicitly.
- Merge headings that are split over two lines, and remove repeated ones ("How We Work How We Work").

### 7.5 Tables
- Split large tables into row groups, **repeating the header row** in each part (current maximum is about 1,250 tokens in the Amazon report).
- Embed a short text summary instead of raw markdown (or add it as a second, table-summary chunk). Numeric tables embed poorly.
- Add the paragraph that references the table ("as shown in Table 3 …") to its metadata.

### 7.6 Figures
Many WattBot answers come from charts. `scripts/parse_pdfs.py` already saves figure images to `documents/figures/<doc_id>/`. A vision model could write a figure description to emit as a `figure` chunk with its caption, page and image path.

### 7.7 Retrieval-side structure
- **Small-to-big / parent retrieval:** retrieve on small chunks, then send the parent section (or `neighbors`) to the LLM.
- **Hybrid search:** keep a BM25 index alongside the embeddings. Many questions hinge on exact numbers and model names ("JetMoE-8B", "3 Wh").
- **Metadata filtering:** by document type (paper/report), year and peer-review status, all available from `metadata.csv`.

### 7.8 Chunker refinements
- Semantic chunking: split a long section where adjacent-sentence embedding similarity drops.
- Merge a very short section into its sibling instead of producing a tiny chunk. `MinTokens` currently drops chunks under about 4 words.
- Make the sentence splitter smarter about abbreviations ("et al.", "Fig. 3", "e.g.") and decimals.
- Clean up Docling spacing artifacts (runs of spaces in justified text: `"The  rest  of  the  paper"`).
- Remove near-duplicate chunks (reports repeat the same stats on several pages).

### 7.9 Operational
- Include a config hash in the output file name and chunk metadata, so different `--max-tokens` runs don't overwrite each other and are traceable.
- Incremental re-chunking: skip documents whose parsed JSON hasn't changed.
- Parse `key_value_items` / `form_items` (currently ignored; empty for the current documents).
- A markdown loader, when the input format changes.

## 8. Current output (5 parsed documents)

| Preset | Chunks | Typical median tokens / doc | Notes |
|---|---|---|---|
| `hybrid` | 649 | 160–290 | text chunks ≤ ~450; tables up to ~1,250 |
| `section` | 475 | 160–670 | |
| `fixed` | 537 | 299 | |

### Baseline retrieval results (TF-IDF, 2026-09-30)

These cover 33 scored questions, 28 of them with locatable evidence.

| Preset | doc_recall@1 | passage_recall@1 | passage_recall@5 | passage_recall@20 | passage_mrr | ctx_tokens@5 |
|---|---|---|---|---|---|---|
| `fixed` | 0.939 | 0.571 | 0.821 | 0.964 | 0.715 | 1,491 |
| `hybrid` | 0.939 | 0.607 | **0.893** | 0.929 | 0.718 | 1,085 |
| `hybrid_raw` | 0.909 | 0.643 | 0.857 | 0.929 | 0.738 | 1,096 |
| `section` | 0.909 | 0.643 | 0.857 | 0.929 | **0.744** | 2,292 |

Reading these results:
- With only 28 passage questions, differences under about 0.07 are one or two questions and are not significant.
- `hybrid` has the best recall@5 at about 70% of `fixed`'s context budget and half of `section`'s.
- The heading prefix (`hybrid` vs `hybrid_raw`) helps doc recall@1 and recall@5, but lowers recall@1 slightly. With TF-IDF, the repeated title words dilute the other terms. Dense embeddings may behave differently.

Question notes:
- **q110:** missed in the top 20 by both `hybrid` and `fixed`.
- **q441:** shows a Docling table-parsing error: cell text spilled into the neighbouring column, so no chunk contains the evidence intact.

Use the per-question CSVs in `documents/eval/` to dig into these cases.

Known limitations:
- The Amazon report is flat: its headings are unnumbered, so each chunk's `heading_path` has one level.
- Token counts are estimates.
- Front matter before the first heading (such as the author list) has an empty `heading_path`.
