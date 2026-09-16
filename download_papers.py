#!/usr/bin/env python3
"""Download PDFs from a CSV of URLs, fetching arXiv metadata when available.

Usage:
    python download_papers.py papers.csv
    python download_papers.py papers.csv --url-column source_url --out-dir pdfs
"""
import argparse
import csv
import hashlib
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([a-zA-Z.\-]*/?\d{4,7}(?:\.\d+)?)(?:v\d+)?(?:\.pdf)?"
)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
HEADERS = {"User-Agent": "paper-fetcher/1.0 (contact: set-your-email-here)"}


def extract_arxiv_id(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    resp = requests.get(ARXIV_API, params={"id_list": arxiv_id}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise ValueError(f"no arXiv entry found for id {arxiv_id}")

    title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
    doi = entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS) or ""
    authors = [a.findtext("atom:name", namespaces=ATOM_NS) for a in entry.findall("atom:author", ATOM_NS)]

    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
    pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"

    return {
        "title": title,
        "authors": "; ".join(a for a in authors if a),
        "abstract": summary,
        "published": published,
        "doi": doi,
        "pdf_url": pdf_url,
    }


def download_pdf(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            chunks = resp.iter_content(chunk_size=8192)
            first_chunk = next(chunks, b"")
            if not first_chunk.startswith(b"%PDF") and "pdf" not in content_type.lower():
                raise ValueError(f"response doesn't look like a PDF (Content-Type: {content_type})")
            with open(dest, "wb") as f:
                f.write(first_chunk)
                for chunk in chunks:
                    f.write(chunk)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def slugify(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"[^\w\-]+", "_", text.strip())
    return text.strip("_")[:maxlen] or "untitled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="input CSV with a URL column")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--out-dir", default="pdfs")
    parser.add_argument("--metadata-out", default="metadata.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("no rows found in input CSV.", file=sys.stderr)
        return

    fieldnames = list(rows[0].keys())
    extra_fields = [
        "source_type", "arxiv_id", "fetched_title", "authors",
        "abstract", "published", "doi", "local_path", "status", "error",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    results = []
    last_arxiv_call = 0.0

    for i, row in enumerate(rows, 1):
        url = (row.get(args.url_column) or "").strip()
        print(f"[{i}/{len(rows)}] {url}")
        row["status"], row["error"] = "pending", ""

        if not url:
            row["status"], row["error"] = "skipped", "empty url"
            results.append(row)
            continue

        arxiv_id = extract_arxiv_id(url)

        try:
            if arxiv_id:
                row["source_type"], row["arxiv_id"] = "arxiv", arxiv_id
                dest = out_dir / f"{arxiv_id.replace('/', '_')}.pdf"

                if dest.exists() and dest.stat().st_size > 0:
                    row["status"] = "skipped_exists"
                else:
                    elapsed = time.monotonic() - last_arxiv_call
                    if elapsed < 3.0:
                        time.sleep(3.0 - elapsed)
                    meta = fetch_arxiv_metadata(arxiv_id)
                    last_arxiv_call = time.monotonic()

                    row["fetched_title"] = meta["title"]
                    row["authors"] = meta["authors"]
                    row["abstract"] = meta["abstract"]
                    row["published"] = meta["published"]
                    row["doi"] = meta["doi"]

                    download_pdf(meta["pdf_url"], dest)
                    row["status"] = "ok"
            else:
                row["source_type"] = "direct"
                name_hint = row.get("title") or Path(urlparse(url).path).stem or url
                filename = f"{slugify(name_hint)}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
                dest = out_dir / filename

                if dest.exists() and dest.stat().st_size > 0:
                    row["status"] = "skipped_exists"
                else:
                    download_pdf(url, dest)
                    row["status"] = "ok"

            row["local_path"] = str(dest)
        except Exception as e:
            row["status"], row["error"] = "failed", str(e)
            print(f"  FAILED: {e}", file=sys.stderr)

        results.append(row)

    with open(args.metadata_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped_exists"))
    print(f"\ndone: {ok}/{len(results)} available locally. metadata written to {args.metadata_out}")


if __name__ == "__main__":
    main()
