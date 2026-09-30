"""Offline tests for packaging and installing corpus snapshots."""
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.chunking import Chunk, ChunkStore, DoclingJsonLoader
from wattbot.snapshot import (
    SnapshotError, build_snapshot, describe, install_snapshot, read_manifest, sha256_file,
)
from wattbot.vectorstore import ChunkVectorStore


def _embed(texts):
    return [[1.0, float(len(t) % 7), float("water" in t)] for t in texts]


def _make_project(root: Path) -> None:
    """A tiny repo layout with every snapshot part present."""
    parsed = root / "documents/parsed_json"
    parsed.mkdir(parents=True)
    doc = {
        "name": "doc1",
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
        "texts": [
            {"self_ref": "#/texts/0", "content_layer": "body", "label": "section_header", "text": "1 Intro",
             "prov": [{"page_no": 1}], "children": []},
            {"self_ref": "#/texts/1", "content_layer": "body", "label": "text", "prov": [{"page_no": 1}],
             "children": [], "text": "Data centers used a lot of water for cooling last year."},
        ],
        "groups": [], "tables": [],
        "pictures": [{"self_ref": "#/pictures/0", "image": {"uri": "data:image/png;base64," + "A" * 5000}}],
        "pages": {"1": {"page_no": 1, "image": {"uri": "data:image/png;base64," + "B" * 5000}}},
    }
    (parsed / "doc1.json").write_text(json.dumps(doc))

    chunk = Chunk("doc1", "Data centers used a lot of water.", ("1 Intro",), [1], "text", [],
                  chunk_id="doc1::hybrid::0000", strategy="hybrid", metadata={"ref_id": "ref2024"})
    ChunkStore([chunk]).save(root / "documents/chunks/hybrid.jsonl")
    ChunkVectorStore("hybrid", path=root / "documents/chroma", model="fake", embed_fn=_embed).index([chunk])

    meta = root / "WattBot2026/metadata_downloaded.csv"
    meta.parent.mkdir(parents=True)
    meta.write_text("id,local_path\nref2024,documents/pdfs/doc1.pdf\n")


def test_build_and_install_round_trip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_project(src)
    archive = tmp_path / "snap.tar.gz"

    manifest = build_snapshot(archive, root=src)
    assert set(manifest["parts"]) == {"parsed_json", "chunks", "chroma", "metadata"}
    assert manifest["parts"]["chunks"] == {"hybrid.jsonl": 1}
    assert manifest["parts"]["chroma"]["collections"] == {"hybrid__fake": 1}
    assert read_manifest(archive) == manifest
    assert "1 documents (figure images stripped)" in describe(manifest)

    install_snapshot(archive, root=dst, expected_sha256=sha256_file(archive))

    installed = json.loads((dst / "documents/parsed_json/doc1.json").read_text())
    assert installed["pictures"][0]["image"] is None and installed["pages"]["1"]["image"] is None
    assert DoclingJsonLoader().load(dst / "documents/parsed_json/doc1.json").blocks  # still loads
    assert ChunkStore.load(dst / "documents/chunks/hybrid.jsonl").get("doc1::hybrid::0000")
    assert (dst / "WattBot2026/metadata_downloaded.csv").exists()

    store = ChunkVectorStore("hybrid", path=dst / "documents/chroma", model="fake", embed_fn=_embed)
    assert store.count() == 1
    assert store.search("water", k=1, instruction=None)[0].metadata["ref_id"] == "ref2024"


def test_empty_chroma_and_missing_parts_are_skipped(tmp_path):
    root = tmp_path / "proj"
    (root / "documents/chunks").mkdir(parents=True)
    ChunkStore([]).save(root / "documents/chunks/hybrid.jsonl")
    ChunkVectorStore("hybrid", path=root / "documents/chroma", model="fake", embed_fn=_embed)  # empty

    manifest = build_snapshot(tmp_path / "s.tar.gz", root=root)
    assert set(manifest["parts"]) == {"chunks"}

    with pytest.raises(SnapshotError, match="nothing to snapshot"):
        build_snapshot(tmp_path / "t.tar.gz", root=tmp_path / "empty")


def test_install_refuses_to_overwrite_without_force(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_project(src)
    archive = tmp_path / "snap.tar.gz"
    build_snapshot(archive, root=src)

    (dst / "documents/chunks").mkdir(parents=True)
    (dst / "documents/chunks/mine.jsonl").write_text("local work\n")
    with pytest.raises(SnapshotError, match="would be replaced"):
        install_snapshot(archive, root=dst)

    install_snapshot(archive, root=dst, force=True)
    backups = list((dst / "documents").glob("chunks.bak-*"))
    assert len(backups) == 1 and (backups[0] / "mine.jsonl").read_text() == "local work\n"
    assert (dst / "documents/chunks/hybrid.jsonl").exists()


def test_install_rejects_bad_checksum_and_unexpected_paths(tmp_path):
    src = tmp_path / "src"
    _make_project(src)
    archive = tmp_path / "snap.tar.gz"
    build_snapshot(archive, root=src, parts=["chunks"])
    with pytest.raises(SnapshotError, match="checksum"):
        install_snapshot(archive, root=tmp_path / "dst", expected_sha256="0" * 64)

    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        for name, data in [("manifest.json", json.dumps({"parts": {"chunks": {}}}).encode()),
                           ("documents/chunks/../../../escape.txt", b"x")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(SnapshotError):
        install_snapshot(evil, root=tmp_path / "dst2")
    assert not (tmp_path / "escape.txt").exists()
