import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.chunking import (
    Block, ChunkStore, DoclingJsonLoader, DropSections, FixedSizeChunker, HybridChunker,
    approx_tokens, build_pipeline,
)
from wattbot.chunking.loaders import HeadingTracker

PARSED_DIR = Path("documents/parsed_json")


def _paths(*headings):
    tracker = HeadingTracker()
    out = []
    for h in headings:
        tracker.push(h)
        out.append(tracker.path)
    return out


def test_heading_levels_roman_and_letter():
    paths = _paths("I. Introduction", "II. Method", "A. Setup", "B. Tuning", "C. Arch", "III. Results", "A. Main")
    assert paths[4] == ("II. Method", "C. Arch")  # "C." is a letter here, not roman 100
    assert paths[5] == ("III. Results",)
    assert paths[6] == ("III. Results", "A. Main")


def test_heading_levels_dotted_and_unnumbered():
    paths = _paths("1 Intro", "2 Model", "2.1 MoE", "2.1.1 Detail", "3. Data", "Server Types", "References")
    assert paths[3] == ("2 Model", "2.1 MoE", "2.1.1 Detail")
    assert paths[5] == ("3. Data", "Server Types")  # unnumbered nests under last numbered heading
    assert paths[6] == ("References",)


def _write_docling(tmp_path):
    def text(i, label, s, page=1, layer="body", parent="#/body"):
        return {"self_ref": f"#/texts/{i}", "parent": {"$ref": parent}, "children": [], "content_layer": layer,
                "label": label, "prov": [{"page_no": page}], "text": s, "orig": s}

    texts = [
        text(0, "section_header", "My Paper"),
        text(1, "page_header", "Journal of Things", layer="furniture"),
        text(2, "section_header", "1 Intro"),
        text(3, "text", "Intro body about the paper's contributions."),
        text(4, "caption", "Table 1: Numbers.", page=2, parent="#/tables/0"),
        text(5, "list_item", "first", parent="#/groups/0"),
        text(6, "section_header", "References", page=3),
        text(7, "text", "[1] Someone et al.", page=3),
        text(8, "text", "chart axis label", page=2, parent="#/pictures/0"),
        text(9, "caption", "Figure 1: A chart.", page=2, parent="#/pictures/0"),
    ]
    doc = {
        "name": "fallback",
        "body": {"children": [{"$ref": r} for r in
                              ["#/texts/0", "#/texts/1", "#/texts/2", "#/texts/3", "#/tables/0",
                               "#/groups/0", "#/pictures/0", "#/texts/6", "#/texts/7"]]},
        "texts": texts,
        "groups": [{"self_ref": "#/groups/0", "content_layer": "body", "label": "list",
                    "children": [{"$ref": "#/texts/5"}]}],
        "tables": [{"self_ref": "#/tables/0", "content_layer": "body", "prov": [{"page_no": 2}],
                    "captions": [{"$ref": "#/texts/4"}], "footnotes": [],
                    "data": {"grid": [[{"text": "a"}, {"text": "b"}], [{"text": "1"}, {"text": "2"}]]}}],
        "pictures": [{"self_ref": "#/pictures/0", "content_layer": "body", "captions": [{"$ref": "#/texts/9"}],
                      "children": [{"$ref": "#/texts/8"}, {"$ref": "#/texts/9"}]}],
    }
    path = tmp_path / "doc1.json"
    path.write_text(json.dumps(doc))
    return path


def test_docling_loader(tmp_path):
    doc = DoclingJsonLoader().load(_write_docling(tmp_path))
    texts = [b.text for b in doc.blocks]

    assert doc.title == "My Paper"
    assert "Journal of Things" not in texts            # furniture dropped
    assert "chart axis label" not in texts             # picture-internal text dropped
    assert "Figure 1: A chart." in texts
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.text.startswith("Table 1: Numbers.") and "| a | b |" in table.text
    assert table.page == 2 and "#/texts/4" in table.source_refs
    assert next(b for b in doc.blocks if b.kind == "list_item").text == "- first"
    assert [b.heading_path for b in doc.blocks if b.text.startswith("Intro body")] == [("1 Intro",)]

    kept = [b.text for b in doc.blocks if DropSections().keep(b)]
    assert "[1] Someone et al." not in kept


def _block(text, path=("S",), kind="text", page=1):
    return Block(doc_id="d", kind=kind, text=text, heading_path=path, page=page)


def test_hybrid_chunker_invariants():
    words = lambda n: " ".join(["word"] * n)
    blocks = ([_block(words(100)) for _ in range(5)]
              + [_block(words(900), kind="table")]
              + [_block(words(500), path=("T",)), _block(words(5), path=("T",))])
    chunks = HybridChunker(max_tokens=400, min_tokens=50).chunk(blocks)

    for c in chunks:
        if c.kind != "table":
            assert approx_tokens(c.text) <= 400 + 50
    assert sum(c.kind == "table" for c in chunks) == 1  # oversized table kept whole
    t_chunks = [c for c in chunks if c.heading_path == ("T",)]
    assert sum(len(c.text.split()) for c in t_chunks) == 505  # nothing from section S leaked in
    assert len(t_chunks) == 2  # 500 words split in two; the 5-word tail merged, not left alone


def test_fixed_chunker_overlap():
    chunks = FixedSizeChunker(max_tokens=13, overlap=3).chunk([_block(" ".join(str(i) for i in range(25)))])
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[-1].text.split()[-1] == "24"


def test_pipeline_ids_and_store(tmp_path):
    chunks = build_pipeline("hybrid").run([_write_docling(tmp_path)])
    assert [c.chunk_id for c in chunks] == [f"doc1::hybrid::{i:04d}" for i in range(len(chunks))]
    assert all(c.embed_text.startswith("My Paper") for c in chunks)

    ChunkStore(chunks).save(tmp_path / "c.jsonl")
    store = ChunkStore.load(tmp_path / "c.jsonl")
    first = store.get("doc1::hybrid::0000")
    assert first.heading_path == ("1 Intro",) and first.pages == [1]
    assert [c.chunk_id for c in store.neighbors("doc1::hybrid::0001", k=1)][0] == "doc1::hybrid::0000"


@pytest.mark.skipif(not (PARSED_DIR / "2109.04459.json").exists(), reason="parsed documents not available")
def test_real_document_spot_checks():
    chunks = build_pipeline("hybrid").run([PARSED_DIR / "2109.04459.json"])
    paths = {c.heading_path for c in chunks}
    assert ("III. Software and Dataflow Optimizations", "A. Model Sparsification") in paths
    assert not any("References" in c.heading_path for c in chunks)
    table = next(c for c in chunks if c.kind == "table")
    assert table.text.startswith("Table 1") and "| Datasets |" in table.text
