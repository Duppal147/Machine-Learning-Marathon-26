# Hybrid Retrieval: Design Notes

Status: evaluation path implemented; fusion results in §7, reranking in §8 · Last updated: 2026-10-12

Now that all 122 corpus documents are parsed and indexed, dense retrieval (Qwen embeddings) scores **below** the offline TF-IDF baseline. The two fail on different questions, though. This document records those results and proposes combining lexical and dense retrieval with reciprocal rank fusion (RRF), which is the "Hybrid search" item in `chunking_design.md` §7.7. For the chunking pipeline and the evaluation harness, see [chunking_design.md](chunking_design.md).

---

## 1. Results that motivate this

Both runs use the same 6,033 `hybrid` chunks (all 122 documents) and the same questions from `WattBot2026/train_QA.csv`. 240 questions are scored, and 182 of them have evidence that can be found verbatim in a chunk.

| Retriever | doc@1 | doc@5 | passage@1 | passage@5 | passage@10 | passage@20 | MRR | ctx_tokens@5 |
|---|---|---|---|---|---|---|---|---|
| Dense (`chroma_qwen3-vl-embedding-8b`) | 0.650 | 0.833 | 0.324 | 0.643 | 0.747 | 0.841 | 0.466 | 1,266 |
| TF-IDF | **0.775** | **0.875** | **0.582** | **0.742** | **0.830** | **0.874** | **0.658** | 1,079 |

The index is healthy. The collection holds exactly 6,033 chunks, and no top-5 hit is a stale ID. Dense misses the gold document entirely, even in the top 20, on 16 questions. TF-IDF misses it on 5.

This confirms the first dense run on 8 documents (`chunking_design.md` §8, 2026-10-04), where dense also trailed TF-IDF on every metric and lost mainly to topical distractors. The absolute numbers aren't comparable, though: with 122 documents there are many more plausible wrong matches.

### 1.1 The two retrievers fail on different questions

Overlap of passage@5 hits, over the 182 questions with located evidence:

| | Questions |
|---|---|
| Both hit | 94 |
| Only TF-IDF hits | 41 |
| Only dense hits | 23 |
| Neither hits | 24 |

If a combined retriever kept every hit from both, passage@5 would be (94 + 41 + 23) / 182 = **0.87**, up from 0.74 for the better single retriever.

### 1.2 Where dense loses: exact facts on crowded topics

Most of TF-IDF's wins are questions with specific anchor words or numbers. Dense returns a chunk on the same topic from the wrong document or year:

- **q107** ("average global data center PUE in 2023"): dense ranks first a chunk saying "the US national datacenter average in 2018 was 1.58". The topic, wording and even the number match, but the scope and year are wrong. The gold chunk is at rank 13.
- **q105** (US data center energy growth 2010–2014): dense returns the 2024 US Data Center Energy Usage Report, which starts in 2014. The gold document `wu2021b` isn't in the top 20.
- **q025, q037, q092:** these hinge on rare terms ("Power Purchase Agreements", "Illinois water commission", "hyperscale … billions of gallons"). TF-IDF puts all of them at rank 1.

The corpus is dense with near-duplicate topics (data-center energy, water, PUE, emissions), so exact-term matching is what separates the right passage from its neighbours. The train questions also tend to reuse the evidence wording, which favours lexical retrieval somewhat (see §6).

### 1.3 Where dense wins: questions that need reasoning

passage@5 by the question flags in `train_QA.csv`, counting only questions with located evidence. The samples are small:

| Flag | Questions | Dense | TF-IDF |
|---|---|---|---|
| Math | 11 | **0.91** | 0.64 |
| CrossPaper | 7 | **0.57** | 0.29 |
| Table | 13 | 0.92 | 0.85 |
| Figure | 8 | 0.88 | 0.88 |

Dense is at least as good on every flagged subset and better on Math and CrossPaper, which are the questions whose wording is furthest from the evidence. Switching to lexical-only retrieval would give these up.

## 2. Goals

1. Get the best of both retrievers: aim for passage@5 and MRR at or above TF-IDF, without losing dense's Math and CrossPaper wins.
2. Need no new embedding and no new service. Fusion reuses the existing Chroma index and an in-memory lexical index.
3. Keep fitting into the existing `Retriever` protocol in `src/wattbot/eval/retrieval.py`, so the comparison runs in the current harness.
4. Add no new dependencies. CONTRIBUTING.md asks before `pyproject.toml` changes, and everything needed is already in numpy, scipy and scikit-learn.

## 3. Proposal

```
                      ┌─► lexical (BM25 / TF-IDF) ─► top-N ─┐
question ─────────────┤                                     ├─► RRF ─► top-k chunks
                      └─► dense (Chroma, Qwen)    ─► top-N ─┘
```

### 3.1 Reciprocal rank fusion

Each retriever returns its top `N` chunks (default `N = 50`). Each chunk then gets a score

```
score(c) = Σ_r  w_r / (k_rrf + rank_r(c))        # rank is 1-based; the term is 0 if c isn't in r's top N
```

and the fused list is sorted by score. Defaults: `k_rrf = 60` (the standard value) and `w_r = 1`.

Why RRF rather than adding up the raw scores:
- **Raw scores aren't comparable.** Cosine similarities cluster in a narrow band (about 0.3–0.7), while TF-IDF and BM25 scores are unbounded and vary with the query. Summing them would need per-query normalization, which adds tuning and is fragile.
- **RRF only uses ranks.** It has one parameter it isn't very sensitive to, and it's well established as a strong default for combining lexical and dense retrieval.
- **It rewards agreement.** A chunk ranked moderately by both retrievers beats one ranked highly by only one. That's the behaviour we want for q107-style near-misses.

### 3.2 A BM25 retriever alongside TF-IDF

Add a `BM25Retriever` next to `TfidfRetriever`. BM25 is the usual lexical partner in hybrid search. Its term-frequency saturation and length normalization should suit our very uneven chunk sizes (tables up to about 2,700 tokens) better than TF-IDF does.

- Tokenize with sklearn `CountVectorizer`, using the same English stop words as `TfidfRetriever` and unigrams only, then score with the standard BM25 formula (`k1 = 1.2`, `b = 0.75`) as a sparse matrix product. That's about 30 lines and needs no `rank_bm25` dependency.
- Like TF-IDF, fit on `chunk.embed_text`, which includes the heading prefix.
- Keep `TfidfRetriever`. Fusion takes any lexical retriever, so we compare `rrf(tfidf, dense)` against `rrf(bm25, dense)` and keep whichever wins.

### 3.3 Code shape

In `src/wattbot/eval/retrieval.py`:

```python
class BM25Retriever:
    name = "bm25"
    def fit(self, chunks): ...                      # CountVectorizer -> doc lengths, idf
    def search(self, queries, k): ...               # BM25 scores -> top-k indices

class FusionRetriever:
    """Reciprocal rank fusion over retrievers that share the same fitted chunk list."""
    def __init__(self, retrievers, depth=50, k_rrf=60, weights=None): ...
    name -> "rrf_" + "+".join(r.name for r in retrievers)
    def fit(self, chunks): for r in retrievers: r.fit(chunks)
    def search(self, queries, k): fuse each r.search(queries, depth), return top-k
```

`FusionRetriever` works on the chunk indices the other retrievers already return. `ChromaRetriever.search` already maps hits back to positions in the fitted list and drops stale IDs, so fusion needs no Chroma-specific code.

In `scripts/eval_retrieval.py`:
- Extend `--retriever` to accept `bm25` and `hybrid`. Add `--lexical tfidf|bm25` (default `bm25`), `--rrf-k` and `--fusion-depth`.
- The per-question CSV naming stays as it is, e.g. `documents/eval/hybrid__rrf_bm25+chroma_qwen3-vl-embedding-8b.csv`.

### 3.4 Later: the production search path

The evaluation path is enough to decide whether to adopt this. If fusion wins:
- Move the retrievers from `wattbot/eval/` into a `wattbot/retrieval.py` module, which the eval harness then imports.
- Add `--mode dense|lexical|hybrid` to `scripts/vector_index.py search`. Build the lexical index in memory from `documents/chunks/<strategy>.jsonl` at startup; 6k chunks take about a second, so nothing needs persisting.
- Support metadata filters (`--ref-id`) on the lexical side by masking chunks before ranking, mirroring Chroma's `where`.

## 4. Evaluation plan

Run everything on the `hybrid` chunks over the full corpus. Dense queries keep the query instruction.

| Run | Purpose |
|---|---|
| `tfidf`, `bm25`, `chroma` | Single-retriever baselines (BM25 is new) |
| `rrf(tfidf, chroma)`, `rrf(bm25, chroma)` | The proposal |
| `k_rrf ∈ {10, 30, 60}`, `depth ∈ {20, 50}` for the better pair | Sensitivity check, not a fine-tuning sweep |

**Criteria for adopting fusion as the default retriever:**
1. passage@5 and MRR are at or above the best single retriever, by more than one question (about 0.0055 per question over 182 questions).
2. On the Math and CrossPaper subsets, fused passage@5 isn't more than one question below dense.
3. `ctx_tokens@5` isn't above dense's 1,266.

**Small-sample caution:** with 182 passage questions, a one-question change moves recall by about 0.55 points, and the flagged subsets have 7–13 questions each. Keep the parameter grid above small and fixed, and don't tune `k_rrf` or the weights to the decimal on the same train questions we report.

Record the results in `chunking_design.md` §8 (or a results section here) and the per-question CSVs in `documents/eval/`.

## 5. Alternatives considered

| Option | Why not now |
|---|---|
| Lexical-only (switch to TF-IDF/BM25) | Simplest and wins overall, but gives up dense's Math and CrossPaper advantage (§1.3), which are the harder questions. |
| Weighted sum of normalized scores | Needs per-query score normalization and a tuned weight, with more tuning on a small question set than RRF needs. Worth revisiting only if RRF underperforms. |
| Cross-encoder or LLM reranker over the fused top 20–50 | Probably the biggest further gain, but it adds gateway calls and latency per query, and we haven't confirmed a reranking model on BadgerBrain. It works naturally *on top of* fusion as a next step. |
| Contextual retrieval enricher (`chunking_design.md` §7.3) | Targets weak headings and chunk context, which is a different failure from the exact-term misses in §1.2. It also requires re-embedding all 6k chunks. Complementary; do it later. |
| Query rewriting / HyDE with the chat model | Adds an LLM call per query and mainly helps recall on paraphrased questions, where dense already does well. |

## 6. Risks and open questions

- **Bias in the evaluation:** train questions often reuse the wording of their evidence, which may overstate TF-IDF. The test questions (`test_Q.csv`) may be worded differently. This is part of why fusion is preferable to dropping dense retrieval.
- **Oversized table chunks:** 141 chunks are over the 400-token budget, the largest about 2,700 tokens (a mis-parsed GPU energy table in `2002.07795`). The hybrid chunker never splits tables. Long tables may skew BM25 length normalization and inflate `ctx_tokens@5`. Splitting tables by row groups with a repeated header is a separate change to consider if they show up among fusion's misses.
- **Gold labels that may be too strict:** some dense "misses" may be correct answers from a different document than the one the question cites (q105 is one to check). Doc and passage recall only credit the cited `ref_id`.
- **The 24 questions neither retriever finds** in the top 5 are worth inspecting separately. They may be parsing or table problems rather than retrieval problems.

## 7. Results (2026-10-12)

Implemented as proposed in §3.3, in `src/wattbot/hybrid.py` (BM25 scoring, weighted RRF) and `src/wattbot/eval/retrieval.py` (`BM25Retriever`, `FusionRetriever`, `CachedRetriever`). Differences from the proposal:
- BM25 uses `k1 = 1.5`, and it leaves out chunks that share no term with the query. Their order would be arbitrary, so in a fusion they would add rank credit backed by no evidence.
- `FusionRetriever` takes per-retriever `weights`.
- `scripts/sweep_rrf.py` caches the dense rankings in `documents/eval/cache/`. After one gateway run, every fusion setting can be compared offline.

Same 6,033 `hybrid` chunks and 182 passage questions as §1. Fusion uses `depth = 50` unless noted.

| Retriever | passage@5 | passage@20 | MRR | doc@1 | ctx_tokens@5 | Math p@5 | CrossPaper p@5 |
|---|---|---|---|---|---|---|---|
| TF-IDF | 0.742 | 0.874 | 0.658 | 0.775 | 1,079 | 7/11 | 2/7 |
| **BM25** | **0.830** | 0.907 | **0.696** | 0.812 | 1,405 | 8/11 | 3/7 |
| Dense (Qwen) | 0.632 | 0.841 | 0.467 | 0.650 | 1,266 | 10/11 | 4/7 |
| RRF(TF-IDF, dense), k=60 | 0.753 | — | 0.622 | 0.796 | 1,190 | 9/11 | 4/7 |
| RRF(BM25, dense), k=60 | 0.819 | **0.967** | 0.648 | 0.800 | 1,355 | 9/11 | 5/7 |
| RRF(BM25, dense), k=10 / 30 | 0.830 / 0.819 | — | 0.673 / 0.653 | 0.817 / 0.804 | 1,352 / 1,354 | 9/11 | 5/7 |
| RRF(BM25, dense), k=60, depth 20 | 0.835 | — | 0.649 | 0.800 | 1,351 | 9/11 | 4/7 |
| RRF(BM25, dense ×0.1), k=60 | **0.846** | 0.923 | 0.682 | **0.821** | 1,395 | 8/11 | 3/7 |

Against the adoption criteria in §4:
1. **Not met.** No fusion setting beats BM25's MRR. Dense weight 0.1 gains about three questions at passage@5, but loses MRR.
2. **Met at equal weights** (Math 9/11, CrossPaper 5/7, vs dense's 10/11 and 4/7). It fails at dense weight 0.1.
3. **Not met by any BM25 setting.** BM25 itself already returns longer chunks (1,405 tokens), likely the oversized tables from §6.

Conclusions:
- **BM25 is a clear win over TF-IDF**, alone (about 16 more questions at passage@5) and as the lexical partner in fusion. It should replace TF-IDF as the lexical retriever.
- **Fusion alone doesn't beat BM25 for the top 5.** Equal-weight RRF ranks dense's on-topic but answerless chunks above BM25's correct first hits (30 questions lose their rank-1 answer), while rescuing 11 questions BM25 misses entirely.
- **Its value is recall.** Equal-weight RRF puts the answer in the top 20 for 96.7% of questions, versus 90.7% for BM25. That makes it the candidate pool for the cross-encoder reranker suggested in §5 (results in §8).
- Following §4's small-sample caution, the weight sweep is reported for understanding, not as a tuned default. Dense-only results vary by about one question between runs (0.632–0.643).

## 8. Reranking the fused top 20 (2026-10-12)

`CrossEncoderReranker` (in `src/wattbot/eval/retrieval.py`) takes any retriever's top `candidates` chunks and reorders them by a cross-encoder score. The cross-encoder reads the question and each chunk's `embed_text` together. Chunks below the candidate cutoff keep their base order. The reranker runs locally with sentence-transformers, so it needs no gateway call. Pair scores are cached in `documents/eval/rerank_cache/`. Run it with `scripts/eval_rerank.py`.

Same 182 passage questions. The base is equal-weight RRF(BM25, dense) with 20 candidates, unless noted.

| Setup | passage@1 | passage@5 | passage@20 | MRR | doc@1 | ctx_tokens@5 |
|---|---|---|---|---|---|---|
| No reranking (k_rrf=60) | 0.516 | 0.819 | 0.967 | 0.648 | 0.800 | 1,355 |
| + `OpenScholar/OpenScholar_Reranker` (k_rrf=20) | 0.571 | 0.868 | 0.967 | 0.692 | 0.800 | 1,463 |
| + `BAAI/bge-reranker-base` (k_rrf=20 or 60) | 0.692 | 0.923 | 0.967 | 0.795 | 0.875 | ~1,465 |
| **+ `BAAI/bge-reranker-large` (k_rrf=60)** | **0.720** | **0.934** | 0.967 | **0.814** | **0.908** | 1,432 |
| + `bge-reranker-large`, 30 / 50 candidates (k_rrf=20) | 0.720 / 0.725 | 0.929 / 0.929 | 0.967 / 0.956 | 0.811 / 0.812 | 0.904 / 0.900 | ~1,450 |
| BM25 only + `bge-reranker-large` | 0.703 | 0.901 | 0.907 | 0.788 | 0.896 | 1,455 |

Against the adoption criteria in §4, the best setup (standard RRF, top 20, `bge-reranker-large`):
1. **Met.** passage@5 is 0.934 vs BM25's 0.830 (about 19 questions), and MRR is 0.814 vs 0.696.
2. **Met.** Math 10/11 and CrossPaper 4/7, the same as dense. All 13 Table questions are in the top 5.
3. **Not met.** `ctx_tokens@5` is 1,432 vs dense's 1,266. That is close to BM25's own 1,405, and it probably comes from the oversized tables in §6.

Findings:
- **Reranking is the largest single gain so far.** It works because fusion already puts the answer in the top 20 for 96.7% of questions. Reranking BM25's top 20 alone reaches only 0.901.
- **`k_rrf` stops mattering** once the reranker reorders the candidates: k=20 and k=60 give the same results. So the standard `k_rrf = 60` with equal weights is used. Nothing in this setup was tuned on the train questions.
- **More candidates don't help.** 30 or 50 candidates add distractors and can push the answer out of the top 20.
- **The OpenScholar reranker is worse** despite being tuned on scientific papers. It was trained to judge whether a passage helps a literature review on the topic, which favours the same on-topic distractors dense retrieval returns. WattBot needs whether a passage contains a specific fact.
- **Qwen remains the better dense partner.** A local BERT bi-encoder (`BAAI/bge-base-en-v1.5`, `SentenceTransformerRetriever`) scores lower than Qwen alone (passage@5 0.577 vs 0.632) and in fusion. Dense trailing BM25 therefore isn't specific to Qwen.

**Recommended search for answer generation:** RRF(BM25, Qwen dense) with equal weights and `k_rrf = 60`, then rerank the top 20 with `bge-reranker-large` and pass the top 5 on.

**Dependency:** the reranker and the local dense model need `sentence-transformers`. That isn't a project dependency yet, so for now run with `uv run --with sentence-transformers`. `OpenScholar_Reranker` also needs `--with sentencepiece`. The model downloads on first use (`bge-reranker-large` is about 2.1 GB).

