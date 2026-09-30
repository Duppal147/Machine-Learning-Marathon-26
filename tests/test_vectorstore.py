"""Offline tests for the Chroma chunk index; a fake embedder stands in for the gateway."""
import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.chunking import Chunk
from wattbot.vectorstore import ChunkVectorStore, call_with_retry, collection_name, format_query

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words are close. Records every call."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    @property
    def embedded(self) -> int:
        return sum(len(c) for c in self.calls)


def _chunk(i, ref, text, pages=(1,), heading=("1 Intro",)):
    return Chunk(doc_id=f"doc_{ref}", text=text, heading_path=heading, pages=list(pages), kind="text",
                 source_refs=[], chunk_id=f"doc_{ref}::hybrid::{i:04d}", strategy="hybrid",
                 metadata={"ref_id": ref, "year": "2024", "title": f"Title {ref}"})


CHUNKS = [
    _chunk(0, "water2024", "data centers consumed water for cooling in 2023", pages=(3, 4),
           heading=("4. Water", "Direct Water Consumption")),
    _chunk(1, "water2024", "electricity generation and grid carbon intensity"),
    _chunk(2, "gpu2023", "training large language models on gpu clusters uses energy"),
]


@pytest.fixture
def store(tmp_path):
    return ChunkVectorStore("hybrid", path=tmp_path / "chroma", model="fake-model", embed_fn=FakeEmbedder())


def test_collection_name_is_chroma_safe():
    assert collection_name("hybrid", "qwen3-vl-embedding-8b") == "hybrid__qwen3-vl-embedding-8b"
    assert collection_name("hybrid", "org/model:v1") == "hybrid__org-model-v1"


def test_query_instruction_format():
    assert format_query("How much?", "Find it") == "Instruct: Find it\nQuery: How much?"
    assert format_query("How much?", None) == "How much?"


def test_index_and_search_round_trip(store):
    stats = store.index(CHUNKS)
    assert (stats.embedded, stats.skipped, store.count()) == (3, 0, 3)

    hit = store.search("water consumed for cooling by data centers", k=3, instruction=None)[0]
    assert hit.chunk_id == "doc_water2024::hybrid::0000"
    assert hit.text == CHUNKS[0].text
    assert hit.metadata["ref_id"] == "water2024"
    assert hit.metadata["pages"] == "3,4" and hit.metadata["page_start"] == 3
    assert hit.metadata["heading_path"] == "4. Water > Direct Water Consumption"
    assert 0.0 < hit.score <= 1.0


def test_index_is_incremental(store):
    store.index(CHUNKS)
    embedder = store._embed_fn

    again = store.index(CHUNKS)
    assert (again.embedded, again.skipped) == (0, 3)

    changed = [CHUNKS[0], CHUNKS[1], _chunk(2, "gpu2023", "inference energy per query on gpus")]
    before = embedder.embedded
    stats = store.index(changed)
    assert (stats.embedded, stats.skipped) == (1, 2)
    assert embedder.embedded - before == 1


def test_embed_text_not_text_is_embedded(store):
    chunk = _chunk(0, "water2024", "raw passage")
    chunk.embed_text = "Title > Section\n\nraw passage"
    store.index([chunk])
    assert store._embed_fn.calls[0] == ["Title > Section\n\nraw passage"]
    assert store.search("raw passage", k=1, instruction=None)[0].text == "raw passage"


def test_prune_and_where_filter(store):
    store.index(CHUNKS)
    hits = store.search("energy", k=3, where={"ref_id": "gpu2023"}, instruction=None)
    assert [h.metadata["ref_id"] for h in hits] == ["gpu2023"]

    stats = store.index(CHUNKS[:2], prune=True)
    assert stats.deleted == 1 and store.count() == 2


def test_persists_and_reset(tmp_path):
    ChunkVectorStore("hybrid", path=tmp_path, model="m", embed_fn=FakeEmbedder()).index(CHUNKS)
    reopened = ChunkVectorStore("hybrid", path=tmp_path, model="m", embed_fn=FakeEmbedder())
    assert reopened.count() == 3
    assert reopened.index(CHUNKS).embedded == 0
    reopened.reset()
    assert reopened.count() == 0
    # a different model is a different collection
    assert ChunkVectorStore("hybrid", path=tmp_path, model="other", embed_fn=FakeEmbedder()).count() == 0


def test_retry_on_transient_errors_only():
    sleeps, calls = [], []

    def flaky(x):
        calls.append(x)
        if len(calls) < 3:
            raise ConnectionError("gateway reset")
        return "ok"

    assert call_with_retry(flaky, "a", sleep=sleeps.append) == "ok"
    assert sleeps == [5.0, 10.0]

    def broken(x):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        call_with_retry(broken, "a", sleep=sleeps.append)
    with pytest.raises(ConnectionError):
        call_with_retry(lambda x: (_ for _ in ()).throw(ConnectionError()), "a", sleep=lambda s: None)
