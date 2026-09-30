"""Package and install shareable snapshots of the processed corpus.

A snapshot is one .tar.gz holding what is slow or costly to rebuild:

    documents/parsed_json/   Docling output (hours of parsing), embedded figure images stripped
    documents/chunks/        chunk JSONL files
    documents/chroma/        the ChromaDB index (gateway embedding calls)
    WattBot2026/metadata_downloaded.csv   file stem -> ref_id mapping used when re-chunking
    manifest.json            what's inside, and the chromadb version / embedding model that wrote it

Installing extracts it into the repo root. Existing local data is only replaced with
`force=True`, and is moved to a timestamped backup rather than deleted.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

# part name -> path relative to the repo root
PARTS = {
    "parsed_json": Path("documents/parsed_json"),
    "chunks": Path("documents/chunks"),
    "chroma": Path(config.CHROMA_DIR),
    "metadata": Path("WattBot2026/metadata_downloaded.csv"),
}
MANIFEST = "manifest.json"
TAG_PREFIX = "index-"
ASSET_PREFIX = "wattbot-index-"


class SnapshotError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_docling_images(doc: dict) -> dict:
    """Drop base64 page/figure images (~90% of the JSON size); the chunker never reads them
    and parse_pdfs.py saves figures as PNGs separately."""
    for picture in doc.get("pictures", []):
        picture["image"] = None
    for page in doc.get("pages", {}).values():
        page["image"] = None
    return doc


def _chroma_counts(chroma_dir: Path) -> dict[str, int]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir),
                                       settings=chromadb.Settings(anonymized_telemetry=False))
    return {c.name: c.count() for c in client.list_collections()}


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def build_snapshot(out_path: Path, root: Path = Path("."), parts: list[str] | None = None,
                   strip_images: bool = True) -> dict:
    """Write the snapshot archive to out_path and return its manifest.

    Parts that don't exist locally are skipped; a Chroma index with no vectors is skipped too.
    """
    import chromadb

    root = Path(root)
    wanted = parts or list(PARTS)
    unknown = set(wanted) - set(PARTS)
    if unknown:
        raise SnapshotError(f"unknown parts {sorted(unknown)}; choose from {sorted(PARTS)}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "chromadb_version": chromadb.__version__,
        "embedding_model": config.EMBEDDING_MODEL,
        "images_stripped": strip_images,
        "parts": {},
    }
    included = []
    for name in wanted:
        path = root / PARTS[name]
        if not path.exists() or (path.is_dir() and not any(path.iterdir())):
            continue
        if name == "parsed_json":
            manifest["parts"][name] = {"documents": len(list(path.glob("*.json")))}
        elif name == "chunks":
            manifest["parts"][name] = {
                f.name: sum(1 for line in open(f) if line.strip()) for f in sorted(path.glob("*.jsonl"))}
        elif name == "chroma":
            counts = _chroma_counts(path)
            if not any(counts.values()):
                continue  # nothing embedded yet; don't ship an empty index
            manifest["parts"][name] = {"collections": counts}
        else:
            manifest["parts"][name] = {"file": str(PARTS[name])}
        included.append(name)
    if not included:
        raise SnapshotError("nothing to snapshot: none of the parts exist locally")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        _add_bytes(tar, MANIFEST, json.dumps(manifest, indent=2).encode())
        for name in included:
            path = root / PARTS[name]
            if name == "parsed_json" and strip_images:
                for f in sorted(path.glob("*.json")):
                    doc = strip_docling_images(json.loads(f.read_text()))
                    _add_bytes(tar, f"{PARTS[name].as_posix()}/{f.name}", json.dumps(doc).encode())
            else:
                tar.add(path, arcname=PARTS[name].as_posix(),
                        filter=lambda info: None if "__pycache__" in info.name else info)
    return manifest


def read_manifest(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            return json.load(tar.extractfile(MANIFEST))
        except KeyError:
            raise SnapshotError(f"{archive} has no {MANIFEST}; not a WattBot index snapshot") from None


def install_snapshot(archive: Path, root: Path = Path("."), force: bool = False,
                     expected_sha256: str | None = None) -> dict:
    """Extract a snapshot into root. Refuses to replace existing local data unless force=True,
    in which case the existing folders/files are moved aside to *.bak-<timestamp>."""
    archive, root = Path(archive), Path(root)
    if expected_sha256 and sha256_file(archive) != expected_sha256:
        raise SnapshotError(f"checksum mismatch for {archive}; download may be corrupt")

    manifest = read_manifest(archive)
    allowed = {MANIFEST} | {PARTS[name].as_posix() for name in manifest["parts"]}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            unsafe = m.name.startswith("/") or ".." in Path(m.name).parts
            if unsafe or not any(m.name == a or m.name.startswith(a + "/") for a in allowed):
                raise SnapshotError(f"unexpected path in snapshot: {m.name}")

        targets = [root / PARTS[name] for name in manifest["parts"]]
        existing = [t for t in targets if t.exists() and (t.is_file() or any(t.iterdir()))]
        if existing and not force:
            raise SnapshotError(
                "local data would be replaced: " + ", ".join(str(t) for t in existing)
                + " (rerun with --force to move it aside to a .bak folder first)")
        stamp = timestamp()
        for t in existing:
            t.rename(t.with_name(f"{t.name}.bak-{stamp}"))
        for t in targets:  # empty leftovers (e.g. an empty chroma dir) would block extraction
            if t.is_dir() and not any(t.iterdir()):
                shutil.rmtree(t)

        try:  # the "data" filter also blocks links/devices/escaping paths
            tar.extractall(root, members=[m for m in members if m.name != MANIFEST], filter="data")
        except tarfile.FilterError as error:
            raise SnapshotError(f"unsafe entry in snapshot: {error}") from error
    return manifest


def describe(manifest: dict) -> str:
    """Human-readable summary, used for release notes and CLI output."""
    lines = [f"created {manifest['created_at']} from commit {manifest['git_commit']}",
             f"embedding model {manifest['embedding_model']}, chromadb {manifest['chromadb_version']}"]
    parts = manifest["parts"]
    if "parsed_json" in parts:
        lines.append(f"parsed_json: {parts['parsed_json']['documents']} documents"
                     + (" (figure images stripped)" if manifest.get("images_stripped") else ""))
    if "chunks" in parts:
        lines.append("chunks: " + ", ".join(f"{k} ({v} chunks)" for k, v in parts["chunks"].items()))
    if "chroma" in parts:
        lines.append("chroma: " + ", ".join(f"{k} ({v} vectors)" for k, v in parts["chroma"]["collections"].items()))
    if "metadata" in parts:
        lines.append(f"metadata: {parts['metadata']['file']}")
    return "\n".join(lines)
