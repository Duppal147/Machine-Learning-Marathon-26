#!/usr/bin/env python3
"""Download and install a shared corpus snapshot (parsed JSON, chunks, Chroma index).

Usage:
    python scripts/fetch_index.py                    # latest snapshot release
    python scripts/fetch_index.py --list             # show available snapshots
    python scripts/fetch_index.py --tag index-20261001-120000
    python scripts/fetch_index.py --archive path/to/wattbot-index-....tar.gz   # already downloaded
    python scripts/fetch_index.py --force            # replace local data (moved aside to *.bak-<ts>)

Snapshots are published with scripts/publish_index.py as GitHub Releases; no GitHub login
is needed to download them (set GITHUB_TOKEN if you hit the anonymous API rate limit).
"""
import argparse
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.snapshot import ASSET_PREFIX, TAG_PREFIX, SnapshotError, describe, install_snapshot

DEFAULT_REPO = os.environ.get("WATTBOT_INDEX_REPO", "Duppal147/Machine-Learning-Marathon-26")
SNAPSHOT_DIR = Path("documents/snapshots")


def _headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    return {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}


def list_snapshots(repo: str) -> list[dict]:
    resp = requests.get(f"https://api.github.com/repos/{repo}/releases", params={"per_page": 50},
                        headers=_headers(), timeout=30)
    resp.raise_for_status()
    return [r for r in resp.json()
            if r["tag_name"].startswith(TAG_PREFIX) and not r["draft"] and not r["prerelease"]]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={k: v for k, v in _headers().items() if k != "Accept"},
                      stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for block in resp.iter_content(1 << 20):
                f.write(block)
    return dest


def fetch_release(repo: str, tag: str | None) -> tuple[Path, str]:
    releases = list_snapshots(repo)
    if not releases:
        raise SnapshotError(f"no snapshot releases ({TAG_PREFIX}*) found on {repo}")
    release = releases[0] if tag is None else next((r for r in releases if r["tag_name"] == tag), None)
    if release is None:
        raise SnapshotError(f"release {tag} not found on {repo}")

    assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
    archive_name = next((n for n in assets if n.startswith(ASSET_PREFIX) and n.endswith(".tar.gz")), None)
    if archive_name is None or f"{archive_name}.sha256" not in assets:
        raise SnapshotError(f"release {release['tag_name']} is missing its archive or checksum")

    print(f"downloading {release['tag_name']}: {archive_name} ...")
    archive = download(assets[archive_name], SNAPSHOT_DIR / archive_name)
    checksum = requests.get(assets[f"{archive_name}.sha256"], timeout=30).text.split()[0]
    return archive, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", help="snapshot release tag (default: latest)")
    parser.add_argument("--archive", type=Path, help="install a local archive instead of downloading")
    parser.add_argument("--force", action="store_true", help="replace existing local data (kept as *.bak-<ts>)")
    parser.add_argument("--list", action="store_true", help="list available snapshots and exit")
    args = parser.parse_args()

    try:
        if args.list:
            releases = list_snapshots(args.repo)
            if not releases:
                print(f"no snapshot releases ({TAG_PREFIX}*) on {args.repo} yet")
            for r in releases:
                size = sum(a["size"] for a in r["assets"] if a["name"].endswith(".tar.gz")) / 1e6
                print(f"{r['tag_name']}  {r['published_at']}  {size:.1f} MB")
            return
        if args.archive:
            archive, checksum = args.archive, None
            sha_file = args.archive.with_name(args.archive.name + ".sha256")
            if sha_file.exists():
                checksum = sha_file.read_text().split()[0]
        else:
            archive, checksum = fetch_release(args.repo, args.tag)
        manifest = install_snapshot(archive, force=args.force, expected_sha256=checksum)
    except (SnapshotError, requests.RequestException) as error:
        sys.exit(f"error: {error}")

    print(f"installed {archive}\n{describe(manifest)}")


if __name__ == "__main__":
    main()
