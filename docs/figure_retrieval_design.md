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
2. **Figure chunks.** `src/wattbot/figures.py` renders each reading as text (caption on its own line, then axes, legend, data table, summary). The loader emits it as an extra `figure` block, and `HybridChunker` turns it into its own chunk, placed after the section's text. The caption also stays in the text flow, so **the text chunks are byte-for-byte those of `hybrid`** (a test checks this, and so did a run over all 122 documents: 6,033 text chunks plus the figure chunks). The chunk's `metadata["figure_index"]` points back to the image. A picture with no reading (or one read as a logo) just has no figure chunk.
3. **Indexing.** Figure chunks go through the same embedding, Chroma, BM25, RRF and reranker code as text chunks. The existing `hybrid`, `hybrid_raw`, `section` and `fixed` strategies are unchanged: their output is identical to the published snapshot.

   An earlier version replaced the caption with the figure chunk and let it interrupt the text flow, as tables do. That split 594 of the 6,033 text chunks and added 158 more, and cost about 3 points of passage recall@5 with dense retrieval. Making figure chunks purely additive removed that part; see §4 for what remains.
4. **Answering.** Values extracted at indexing time can be misread, so `scripts/ask_figures.py` retrieves figure chunks and then sends the top charts' original images plus the question back to the vision model, asking for the exact printed value. A read that runs out of tokens is retried once at double the budget.

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

`scripts/eval_figures_e2e.py` scores the whole path on the numeric chart questions: it retrieves charts, reads the top `--reads` images with the vision model, and counts a question answered when a number on the first line of a reply equals `answer_value` at its precision (28.5 matches 28.54, not 28.4; 2007 matches only 2007; `--tolerance 0.01` additionally accepts 1% for estimated values). `strict_answer@k` also requires the chart to be from a gold document, so a coincidence in an unrelated chart doesn't count, and `--oracle-doc` searches only the gold document to separate reading errors from retrieval errors. `--prompt default scope` compares two answer prompts on the same retrieved charts (retrieval runs once, so it can't vary between the arms); `scope` is an answer prompt that first makes the model check that the chart's subject, entity, period and metric match the question, and reply "not in figure" otherwise. It targets the failure where a chart on a related topic (Fig. 6 instead of Fig. 8 for a Facebook energy question) has a plausible number on it and the model answers confidently. Replies are cached per prompt. Check `documents/eval/figures_e2e__<strategy>.csv`: a unit difference from `answer_value` shows up there as a wrong number.

**Run-to-run variation.** The gateway's embedding of one text differs slightly between calls (`check_text_embeddings.py` accepts cosine >= 0.9999), which is enough to reorder close results: the same q001 search returned the gold chart at rank 1 in one run and outside the top 3 in another. `ask_figures.py` and `eval_figures_e2e.py` therefore cache query vectors in `documents/eval/query_embeddings.json` (`--fresh-queries` opts out), so repeated runs retrieve the same charts. `eval_figures.py` and `eval_retrieval.py` still embed the questions fresh each run, so their rankings can move slightly between runs; compare strategies within one invocation, as those scripts do.

Samples are tiny (14 questions score `figure_recall`, 10 score `value_recall`; one question is 7–10 points), so treat differences of a question or two as noise.

**Does adding figure chunks hurt text questions?** Compare passage recall with `scripts/eval_retrieval.py --retriever hybrid --strategies hybrid hybrid_fig`. With the original, text-splitting design the dense hybrid retriever lost 2.8 points at passage@5 (0.819 to 0.791) and 3.3 at @10. Offline BM25 on `hybrid_figlabels` shows the same effect split in two: the splitting cost about 1 point at @10, and the extra chunks competing with the right text chunk cost about 1 point at @5 that the additive design does not remove. Closing that gap is a retrieval-design choice (reserve slots for figure chunks, give them a separate lane, or rerank), not a chunking one.

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

# 4. score the whole path on the numeric chart questions (about 3 gateway calls per question)
uv run python scripts/eval_figures_e2e.py --reads 3
uv run python scripts/eval_figures_e2e.py --reads 3 --oracle-doc   # reading accuracy without retrieval errors
uv run python scripts/eval_figures_e2e.py --reads 3 --prompt default scope   # A/B on the same charts

# 5. ask a question end to end
uv run python scripts/ask_figures.py "What operational energy footprint reduction did Facebook achieve over two years?" --ref-id wu2021a --read 1
```

**Teammates can skip step 2.** The readings (about an hour of gateway calls) are a snapshot part, `figure_text`, so `fetch_index.py` installs them. After that, `chunk_docs.py --strategy hybrid_fig` works without running `describe_figures.py`, and the Chroma collection `hybrid_fig__…` comes with the snapshot if its publisher ran `vector_index.py index --strategy hybrid_fig` first. Snapshots still don't include the figure PNGs, which `ask_figures.py` and `review_figures.py` need; write them with `parse_pdfs.py --all --force`. If you re-run `describe_figures.py` after regenerating the PNGs, it re-reads any figure whose image bytes differ from the one the cached reading was made from, so don't run it just to "refresh". To share your readings:

```bash
uv run python scripts/publish_index.py --upload     # includes figure_text and the hybrid_fig index if present
```

Inspect `documents/figure_text/<doc>/figure_NN.json` next to `documents/figures/<doc>/figure_NN.png` for a few charts before running everything: the data table is only as good as the model's reading.

### Writing the pull request description

`scripts/make_pr_body.py` writes `pr_body.md` from measured results, so no number is typed by hand. The three eval scripts save a summary (`documents/eval/summary__*.json`) stamped with the commit that produced it; the generator reads those, counts the readings and figure chunks, checks that `hybrid_fig`'s text chunks are identical to `hybrid`'s, lists the shared files in the diff, and runs the tests. A result that is missing, from another commit, or from uncommitted changes is flagged in the text and listed at the end (`--strict` exits non-zero).

```bash
git status --short                    # commit first, so results are stamped with the commit being proposed
uv run --with pytest python scripts/make_pr_body.py --run     # runs chunking, indexing and the evals, then writes pr_body.md
gh pr create --base main --head <branch> --title "feat: chart reading and figure chunks for retrieval" --body-file pr_body.md
```

Without `--run` it only reads existing results. The manual checks (vision-model check, spot-checking readings, a non-author approval) stay as unticked boxes. Don't commit `pr_body.md`.

## 6. Known limits and next steps

- **Reading quality is unmeasured.** The offline tests use a fake client. The first real run should spot-check 10–20 charts, including a dense multi-series line chart, a stacked bar, and a small-print one, and compare the extracted table with the image.
- **Prompt tuning.** `PROMPT_VERSION` in `figures.py` must be bumped when `FIGURE_PROMPT` changes, so cached readings are redone.
- **Images are downscaled** to 2,048 px on the longest side before upload. Very dense charts may need a higher limit (`MAX_IMAGE_SIDE`).
- **Multi-panel figures** are read as one image. Splitting panels `(a)`, `(b)` is a possible refinement.
- **Tables-as-images** are classified `table` and read like charts. Docling's own table extraction doesn't see them.
- **End-to-end scoring** covers numeric answers only. True/false, category and text answers (and the `Reconcile` and `Corrected` columns of `train_QA.csv`) aren't scored.
