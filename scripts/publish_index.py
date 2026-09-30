#!/usr/bin/env python3
"""Package the processed corpus (parsed JSON, chunks, Chroma index) and publish it as a GitHub Release.

Usage:
    python scripts/publish_index.py                 # build documents/snapshots/wattbot-index-<ts>.tar.gz only
    python scripts/publish_index.py --upload        # ... and publish it as release index-<ts>
    python scripts/publish_index.py --parts parsed_json chunks metadata --upload

Uploading uses the GitHub CLI (`gh auth login`) and needs write access to the repo.
The team repo is public, so published snapshots can be downloaded by anyone.
Teammates install the latest snapshot with scripts/fetch_index.py.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.snapshot import (
    ASSET_PREFIX, PARTS, TAG_PREFIX, SnapshotError, build_snapshot, describe, sha256_file, timestamp,
)

DEFAULT_REPO = os.environ.get("WATTBOT_INDEX_REPO", "Duppal147/Machine-Learning-Marathon-26")
SNAPSHOT_DIR = Path("documents/snapshots")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", choices=sorted(PARTS), help="default: every part that exists")
    parser.add_argument("--keep-images", action="store_true", help="keep base64 figure images in parsed JSON")
    parser.add_argument("--upload", action="store_true", help="create a GitHub Release with the archive")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--notes", default="", help="extra text for the release notes")
    args = parser.parse_args()

    stamp = timestamp()
    archive = SNAPSHOT_DIR / f"{ASSET_PREFIX}{stamp}.tar.gz"
    try:
        manifest = build_snapshot(archive, parts=args.parts, strip_images=not args.keep_images)
    except SnapshotError as error:
        sys.exit(f"error: {error}")

    checksum_file = archive.with_name(archive.name + ".sha256")
    checksum_file.write_text(f"{sha256_file(archive)}  {archive.name}\n")
    summary = describe(manifest)
    print(f"built {archive} ({archive.stat().st_size / 1e6:.1f} MB)\n{summary}")
    if "chroma" not in manifest["parts"]:
        print("note: no embedded Chroma index included (run scripts/vector_index.py index first to share one)")

    if not args.upload:
        print("\nnot uploaded (pass --upload to publish it as a GitHub Release)")
        return

    tag = f"{TAG_PREFIX}{stamp}"
    notes = f"{args.notes}\n\n{summary}\n\nInstall: `python scripts/fetch_index.py --tag {tag}`".strip()
    subprocess.run(["gh", "release", "create", tag, str(archive), str(checksum_file), "--repo", args.repo,
                    "--title", f"Index snapshot {stamp}", "--notes", notes], check=True)
    print(f"published release {tag} on {args.repo}")


if __name__ == "__main__":
    main()
