import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.chunking import Chunk
from wattbot.eval import BM25Retriever, CachedRetriever, ChromaRetriever, FusionRetriever, Question, evaluate
from wattbot.hybrid import Bm25Index, rrf_fuse, tokenize


def test_tokenize_keeps_numbers_and_model_names_drops_stopwords():
    assert tokenize("The JetMoE-8B used 1.9 GWh and 4,096 H100 GPUs") == \
        ["jetmoe", "8b", "used", "1.9", "gwh", "4,096", "h100", "gpus"]


def test_bm25_ranks_exact_rare_term_first():
    texts = [
        "Energy use of large language models during training and inference.",
        "JetMoE-8B was trained on 96 H100 GPUs for two weeks.",
        "Training energy for language models depends on hardware and energy mix.",
    ]
    index = Bm25Index().fit(texts)
    assert index.search(["How many GPUs trained JetMoE-8B?"], k=1) == [[1]]


def test_bm25_rarer_term_weighs_more_and_length_normalizes():
    texts = ["water water", "water carbon", "carbon " + "filler " * 30]
    scores = Bm25Index().fit(texts).scores(["carbon"])[0]
    assert scores[0] == 0.0
    assert scores[1] > scores[2] > 0  # same term frequency, but doc 2 is much longer
    idf = Bm25Index().fit(texts).idf
    vocab = Bm25Index().fit(texts).vectorizer.vocabulary_
    assert idf[vocab["filler"]] > idf[vocab["water"]]  # appears in fewer documents


def test_bm25_scores_shape_and_unknown_query():
    index = Bm25Index().fit(["alpha beta", "gamma delta", "beta gamma"])
    scores = index.scores(["beta", "zzz unknown"])
    assert scores.shape == (2, 3)
    assert np.all(scores[1] == 0)


def test_bm25_search_leaves_out_documents_with_no_shared_term():
    index = Bm25Index().fit(["alpha beta", "gamma delta", "beta gamma"])
    assert sorted(index.search(["beta"], k=3)[0]) == [0, 2]
    assert index.search(["zzz unknown"], k=3) == [[]]


def test_rrf_fuse_rewards_agreement_over_one_strong_rank():
    # 7 is first in one list only; 3 is second in both, so it wins
    assert rrf_fuse([[7, 3, 1], [3, 2, 9]])[0] == 3
    assert rrf_fuse([[7, 3, 1], [3, 2, 9]], limit=2) == [3, 7]


def test_rrf_fuse_handles_missing_items_and_ties_deterministically():
    fused = rrf_fuse([[1, 2], [2]])
    assert fused == [2, 1]
    assert rrf_fuse([[5], [6]]) == [5, 6]  # equal scores keep first-seen order
    assert rrf_fuse([]) == []


def test_rrf_fuse_weights_let_the_stronger_ranking_decide_the_top():
    lexical, dense = [1, 2, 3], [2, 4, 1]
    assert rrf_fuse([lexical, dense])[0] == 2                       # equal weights: agreement wins
    assert rrf_fuse([lexical, dense], weights=[1.0, 0.2])[0] == 1   # lexical leads
    assert 4 in rrf_fuse([lexical, dense], weights=[1.0, 0.2])     # dense-only hit is kept, lower down
    with pytest.raises(ValueError):
        rrf_fuse([lexical, dense], weights=[1.0])


def _chunk(i, ref, text):
    return Chunk("d", text, (), [1], "text", [], chunk_id=f"c{i}", metadata={"ref_id": ref})


def _concept_embedder(texts):
    """Fake dense embedder that knows 'PUE' means power usage effectiveness (no gateway needed)."""
    concepts = [("pue", "power usage effectiveness"), ("water",), ("gpu", "h100")]
    vecs = [[float(any(w in t.lower() for w in group)) for group in concepts] for t in texts]
    return [v if any(v) else [1e-3] * len(concepts) for v in vecs]


def test_fusion_retriever_combines_lexical_and_dense_hits(tmp_path):
    from wattbot.vectorstore import ChunkVectorStore

    chunks = [
        _chunk(0, "c2024", "Water consumption of data centers keeps rising."),
        _chunk(1, "b2024", "JetMoE-8B was trained on 96 H100 GPUs."),
        _chunk(2, "a2024", "The facility reported a power usage effectiveness of 1.10 in 2021."),
    ]
    questions = [
        # shares no word with its evidence: only dense can bridge PUE -> power usage effectiveness
        Question("q1", "What PUE was achieved?", ["a2024"],
                 ["reported a power usage effectiveness of 1.10 in 2021"]),
        # exact model name, which BM25 matches directly
        Question("q2", "How many GPUs were used for JetMoE-8B?", ["b2024"],
                 ["JetMoE-8B was trained on 96 H100 GPUs"]),
    ]
    store = ChunkVectorStore("toy", path=tmp_path, model="fake", embed_fn=_concept_embedder)
    dense = ChromaRetriever("toy", query_instruction=False, store=store)
    hybrid = FusionRetriever([BM25Retriever(), dense])

    summary, results = evaluate(chunks, questions, hybrid, ks=(1, 2))
    assert hybrid.name == "rrf_bm25+chroma_fake_noinstr"
    assert [r.top_chunk_ids[0] for r in results] == ["c2", "c1"]
    assert summary["passage_recall@1"] == 1.0

    bm25_summary, _ = evaluate(chunks, questions, BM25Retriever(), ks=(1,))
    assert bm25_summary["passage_recall@1"] < 1.0  # BM25 alone misses the PUE paraphrase


class _CountingRetriever:
    """Returns a fixed ranking and counts how often it is actually searched."""
    name = "counting"

    def __init__(self, order):
        self.order, self.fits, self.calls = order, 0, 0

    def fit(self, chunks):
        self.fits += 1

    def search(self, queries, k):
        self.calls += 1
        return [self.order[:k] for _ in queries]


def test_cached_retriever_reuses_rankings_across_runs(tmp_path):
    chunks = [_chunk(i, "a2024", f"text {i}") for i in range(4)]
    path = tmp_path / "cache.json"

    inner = _CountingRetriever([2, 0, 3, 1])
    cached = CachedRetriever(inner, path)
    cached.fit(chunks)
    assert cached.search(["q1", "q2"], k=3) == [[2, 0, 3], [2, 0, 3]]
    assert cached.search(["q1"], k=2) == [[2, 0]]           # shallower: served from cache
    assert (inner.fits, inner.calls) == (1, 1)

    again = _CountingRetriever([9, 9, 9, 9])                # a fresh process reads the file
    reloaded = CachedRetriever(again, path)
    reloaded.fit(chunks[1:] + chunks[:1])                   # different order: ids map to new positions
    assert reloaded.search(["q1"], k=3) == [[1, 3, 2]]
    assert (again.fits, again.calls) == (0, 0)              # never touched the inner retriever

    reloaded.fit(chunks[:3])                                # chunk c3 no longer exists
    assert reloaded.search(["q2"], k=3) == [[2, 0]]         # its cached id is dropped
    assert reloaded.name == "counting"


def test_cached_retriever_fetches_deeper_when_k_grows(tmp_path):
    chunks = [_chunk(i, "a2024", f"text {i}") for i in range(4)]
    inner = _CountingRetriever([3, 2, 1, 0])
    cached = CachedRetriever(inner, tmp_path / "c.json")
    cached.fit(chunks)
    cached.search(["q"], k=2)
    assert cached.search(["q"], k=4) == [[3, 2, 1, 0]]
    assert inner.calls == 2


def test_fusion_retriever_name_records_weights_and_k():
    assert FusionRetriever([BM25Retriever(), BM25Retriever()], weights=[1.0, 0.3]).name == "rrf_bm25+bm25x0.3"
    assert FusionRetriever([BM25Retriever()], k_rrf=20).name == "rrf_bm25_k20"

