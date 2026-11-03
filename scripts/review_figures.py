#!/usr/bin/env python3
"""Build an HTML page showing each chart next to its vision-model reading, to check the values by eye.

Usage:
    python scripts/review_figures.py [--n 20] [--seed 0] [--kind line_chart ...] [--doc doc_id ...]
                                     [--out documents/review/figures.html]

Samples readings from documents/figure_text (written by scripts/describe_figures.py) and puts the image
from documents/figures on the left and the text that would be indexed on the right. Open the page in a
browser: `open documents/review/figures.html` on macOS. No gateway needed.

What to check on each card: every printed value is in the data table and matches, estimates are marked
with ~, no series is missing or invented, units are right. Readings without a table (maps, diagrams,
infographics) should at least say what the figure shows. Logos and photos are listed as dropped.
"""
import argparse
import html
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import figures

PARSED_DIR = Path("documents/parsed_json")

STYLE = """
:root { color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --warn:#b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#eee; --muted:#999; --line:#333; --warn:#ff8a80; } }
body { background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1200px; padding:16px; }
.card { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; border-top:1px solid var(--line); padding:16px 0; }
.card img { max-width:100%; height:auto; background:#fff; border:1px solid var(--line); }
.meta { color:var(--muted); font-size:13px; margin-bottom:6px; }
pre { white-space:pre-wrap; word-break:break-word; margin:0; font:13px/1.4 ui-monospace,monospace; }
.bad { color:var(--warn); font-weight:600; }
@media (max-width:760px) { .card { grid-template-columns:1fr; } }
"""


def collect(doc_ids: list[str] | None) -> list[tuple[str, int, dict]]:
    found = []
    for path in sorted(figures.FIGURE_TEXT_DIR.glob("*/figure_*.json")):
        doc_id, index = path.parent.name, int(path.stem.split("_")[1])
        if doc_ids and doc_id not in doc_ids:
            continue
        record = figures.load_record(path)
        if record:
            found.append((doc_id, index, record))
    return found


def card(doc_id: str, index: int, record: dict, caption: str, page, out_dir: Path) -> str:
    result = record["result"]
    image = os.path.relpath(figures.figure_image_path(doc_id, index), out_dir)
    text = figures.render_description(result, caption)
    flag = ' <span class="bad">(reply was not valid JSON)</span>' if result.get("parse_error") else ""
    body = html.escape(text) if text else f"(dropped from the index: read as {html.escape(result.get('kind', '?'))})"
    return (f'<div class="card"><div><div class="meta">{html.escape(doc_id)} · figure_{index:02d} · '
            f'page {page or "?"} · read as {html.escape(result.get("kind", "?"))}{flag}</div>'
            f'<a href="{html.escape(image)}"><img src="{html.escape(image)}" loading="lazy" alt=""></a></div>'
            f'<div><div class="meta">indexed text · {html.escape(record.get("model", "?"))}</div>'
            f'<pre>{body}</pre></div></div>')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="how many readings to show")
    parser.add_argument("--seed", type=int, default=0, help="change to see a different sample")
    parser.add_argument("--kind", nargs="+", help="only these chart kinds, e.g. line_chart stacked_chart")
    parser.add_argument("--doc", nargs="+", help="only these doc_ids")
    parser.add_argument("--out", type=Path, default=Path("documents/review/figures.html"))
    args = parser.parse_args()

    readings = collect(args.doc)
    if not readings:
        sys.exit(f"error: no readings in {figures.FIGURE_TEXT_DIR}; run scripts/describe_figures.py first")
    kinds = Counter(r["result"].get("kind", "?") for _, _, r in readings)
    errors = sum(bool(r["result"].get("parse_error")) for _, _, r in readings)
    pool = [x for x in readings if not args.kind or x[2]["result"].get("kind") in args.kind]
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))

    parsed: dict[str, dict] = {}
    cards = []
    for doc_id, index, record in sample:
        if doc_id not in parsed:
            path = PARSED_DIR / f"{doc_id}.json"
            parsed[doc_id] = json.loads(path.read_text()) if path.exists() else {"pictures": [], "texts": []}
        doc = parsed[doc_id]
        picture = next((p for p in doc["pictures"] if figures.picture_index(p) == index), None)
        caption = figures.picture_caption(doc, picture) if picture else ""
        page = (picture.get("prov") or [{}])[0].get("page_no") if picture else None
        cards.append(card(doc_id, index, record, caption, page, args.out.parent))

    summary = ", ".join(f"{k} {n}" for k, n in kinds.most_common())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>Figure readings</title><style>{STYLE}</style></head>"
        f"<body><h1>Figure readings</h1><p class='meta'>{len(readings)} readings: {html.escape(summary)}. "
        f"Not valid JSON: {errors}. Showing {len(cards)} (seed {args.seed}).</p>{''.join(cards)}</body></html>")
    print(f"{len(readings)} readings ({summary}); {errors} not valid JSON")
    print(f"wrote {len(cards)} cards -> {args.out}   (open it in a browser)")


if __name__ == "__main__":
    main()
