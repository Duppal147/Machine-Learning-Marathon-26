"""Small JSON summaries of evaluation runs, stamped with the commit that produced them.

scripts/make_pr_body.py reads these to write a pull request description, and flags a result that was
produced at a different commit than the one being proposed, so a stale number can't slip into a PR.
"""
from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_state() -> tuple[str, bool]:
    """(short commit hash, working tree has uncommitted changes); ("unknown", False) outside a repo."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def _clean(value):
    """JSON-safe: numpy numbers become plain numbers, NaN becomes null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_summary(path: Path, kind: str, payload: dict) -> None:
    commit, dirty = git_state()
    record = {"kind": kind, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": commit, "git_dirty": dirty, **_clean(payload)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def load_summary(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
