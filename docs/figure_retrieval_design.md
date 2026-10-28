# Figure retrieval: reading charts with a vision model

Many WattBot answers sit in charts, and the text pipeline throws that content away: the loader keeps only a picture's caption (`chunking_design.md` §7.6). This document describes the first pass at making chart values searchable and answerable, and how to compare it with simpler options. For the text pipeline itself see [chunking_design.md](chunking_design.md) and [pipeline_runbook.md](pipeline_runbook.md).

## 1. What the corpus contains

From the 122 parsed documents:

- 1,435 Docling "pictures", of which 986 have a caption and 237 are tiny (under 5,000 pt², mostly icons and logos). Six documents have none.
- The counts are skewed: the Amazon sustainability report has 223 pictures (28 captioned, mostly infographics), the IEA special report 133.
- 625 pictures carry five or more text items that Docling read from the PDF text layer (axis ticks, series names, printed values). The loader currently discards all 28,277 of them.
- 17 train questions are flagged `Figure`. 14 name a figure in `supporting_materials` and 10 have a plain numeric answer.

## 2. Approach

```
parse_pdfs.py ──► documents/figures/<doc>/figure_NN.png
                          │
            describe_figures.py   (vision model, one call per chart, cached)
                          ▼
              documents/figure_text/<doc>/figure_NN.json
                          │
            chunk_docs.py --strategy hybrid_fig     one "figure" chunk per chart
                          ▼
            vector_index.py index --strategy hybrid_fig     (Qwen text embedding, as for text)
                          │
   BM25 + dense (RRF) ──► ask_figures.py ──► the chart image is re-read with the question
```

1. **Reading.** `scripts/describe_figures.py` sends each figure PNG to the vision model with its caption and the labels Docling found. The model returns JSON: chart type, title, axes, legend, a markdown data table, and a short summary. Logos and photos are classified and dropped. Results are cached per image (hash plus prompt version), so re-runs only read what changed.
2. **Figure chunks.** `src/wattbot/figures.py` renders each reading as text (caption on its own line, then axes, legend, data table, summary). The loader emits it as a `figure` block, which `HybridChunker` keeps whole like a table. The chunk's `metadata["figure_index"]` points back to the image. A picture with no reading (or one read as a logo) falls back to caption-only, exactly as before.
3. **Indexing.** Figure chunks go through the same embedding, Chroma, BM25, RRF and reranker code as text chunks. The existing `hybrid`, `hybrid_raw`, `section` and `fixed` strategies are unchanged: their output is identical to the published snapshot.
4. **Answering.** Values extracted at indexing time can be misread, so `scripts/ask_figures.py` retrieves figure chunks and then sends the top chart's original image plus the question back to the vision model, asking for the exact printed value.

## 3. Strategies to compare

| Strategy | Figure chunk text | Needs |
|---|---|---|
| `hybrid` | none (captions only, as today) | nothing |
| `hybrid_figlabels` | caption + labels from the PDF text layer | nothing (no model) |
| `hybrid_fig` | caption + vision-model reading (axes, data table, summary) | `describe_figures.py` |

Planned comparison arms, not built yet: embedding the chart PNG directly with `qwen3-vl-embedding` (`check_vision_model.py --probe-image-embedding` tests whether the gateway accepts images), and page-image late-interaction models (ColPali family), which need a local GPU.

## 4. Evaluation

`scripts/eval_figures.py` scores the 17 `Figure` questions, which the passage metrics in `eval_retrieval.py` cannot (their evidence isn't quotable text). Both metrics only count chunks of a gold document:

- `figure_recall@k`: a top-k chunk is the caption of a figure that `supporting_materials` names. It measures whether the right chart is found.
- `value_recall@k`: a top-k chunk contains the numeric answer. It measures whether the exact value is searchable. A number repeated somewhere unrelated also counts, so read it next to `figure_recall`.

Baseline with BM25 on the full corpus (offline, from the published snapshot's parsed JSON; 14 questions score `figure_recall`, 10 score `value_recall`):

| Strategy | Figure chunks | figure@1 | figure@5 | figure@10 | value@1 | value@5 | value@10 |
|---|---|---|---|---|---|---|---|
| `hybrid` | 0 | 0.214 | 0.429 | 0.571 | 0.400 | 0.500 | 0.500 |
| `hybrid_figlabels` | 631 | 0.286 | 0.429 | 0.571 | 0.400 | 0.600 | 0.600 |

The samples are tiny (one question is 7–10 points), so treat differences of a question or two as noise. `hybrid_fig` has not been run yet: it needs the gateway.

## 5. Running it

On a machine with the VPN and `OPENAI_API_KEY` (see the README), from the repo root:

```bash
# 0. figure PNGs are not in snapshots; write them (about 10 min, no VPN)
uv run python scripts/download_papers.py WattBot2026/metadata.csv --out-dir documents/pdfs --metadata-out WattBot2026/metadata_downloaded.csv
uv run python scripts/parse_pdfs.py --all --force      # skip --force if documents/parsed_json is missing

# 1. check the vision model reads a chart (lists the models the gateway serves)
export VISION_MODEL=...                                # optional if exactly one *vl* chat model is served
uv run python scripts/check_vision_model.py --probe-image-embedding

# 2. read the charts (resumable; start small)
uv run python scripts/describe_figures.py --dry-run
uv run python scripts/describe_figures.py --limit 5
uv run python scripts/describe_figures.py --workers 4

# 3. chunk, embed, evaluate
uv run python scripts/chunk_docs.py --strategy hybrid_fig
uv run python scripts/vector_index.py index --strategy hybrid_fig
uv run python scripts/eval_figures.py --retriever bm25
uv run python scripts/eval_figures.py --retriever hybrid

# 4. ask a question end to end
uv run python scripts/ask_figures.py "What operational energy footprint reduction did Facebook achieve over two years?" --ref-id wu2021a --read 1
```

Inspect `documents/figure_text/<doc>/figure_NN.json` next to `documents/figures/<doc>/figure_NN.png` for a few charts before running everything: the data table is only as good as the model's reading.

## 6. Known limits and next steps

- **Reading quality is unmeasured.** The offline tests use a fake client. The first real run should spot-check 10–20 charts, including a dense multi-series line chart, a stacked bar, and a small-print one, and compare the extracted table with the image.
- **Prompt tuning.** `PROMPT_VERSION` in `figures.py` must be bumped when `FIGURE_PROMPT` changes, so cached readings are redone.
- **Images are downscaled** to 2,048 px on the longest side before upload. Very dense charts may need a higher limit (`MAX_IMAGE_SIDE`).
- **Multi-panel figures** are read as one image. Splitting panels `(a)`, `(b)` is a possible refinement.
- **Tables-as-images** are classified `table` and read like charts. Docling's own table extraction doesn't see them.
- **End-to-end scoring** (numeric answer within tolerance of `answer_value`) isn't built; `ask_figures.py` prints the answer for manual review. The `Reconcile` and `Corrected` columns in `train_QA.csv` have not been looked at.
