#!/usr/bin/env python3
"""Convert PDFs with Docling: structured JSON + extracted figure images.

Usage:
    python scripts/parse_pdfs.py [--all [--force]] [doc_id ...]

doc_id is the PDF filename without ".pdf", e.g. "2501.16548" or
"2023_Amazon_Sustainability_Report_0186f6c3". Explicit doc_ids are always (re-)parsed.
With no doc_ids (or --all), every PDF in documents/pdfs without a parsed JSON is parsed;
--force re-parses those too. A failed document is reported and skipped, so long runs
continue, and JSON is written atomically, so an interrupted run resumes cleanly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

PDF_DIR = Path("documents/pdfs")
JSON_OUT = Path("documents/parsed_json")
FIGURES_OUT = Path("documents/figures")


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def pending_doc_ids(force: bool) -> list[str]:
    doc_ids = sorted(p.stem for p in PDF_DIR.glob("*.pdf"))
    if force:
        return doc_ids
    return [d for d in doc_ids if not (JSON_OUT / f"{d}.json").exists()]


def parse_one(converter: DocumentConverter, doc_id: str) -> None:
    result = converter.convert(str(PDF_DIR / f"{doc_id}.pdf"))
    doc = result.document

    json_path = JSON_OUT / f"{doc_id}.json"
    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(doc.export_to_dict(), indent=2))
    tmp_path.replace(json_path)

    fig_dir = FIGURES_OUT / doc_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_images = 0
    for i, picture in enumerate(doc.pictures):
        try:
            img = picture.get_image(doc)
        except Exception as e:
            print(f"  picture {i} failed: {e}", file=sys.stderr)
            continue
        if img is None:
            continue
        img.save(fig_dir / f"figure_{i:02d}.png")
        n_images += 1

    print(f"  pages={len(doc.pages)} tables={len(doc.tables)} images_extracted={n_images}")
    print(f"  json -> {json_path}")
    print(f"  figures -> {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_ids", nargs="*")
    parser.add_argument("--all", action="store_true", help="parse every PDF that has no parsed JSON yet")
    parser.add_argument("--force", action="store_true", help="with --all: re-parse already parsed PDFs too")
    args = parser.parse_args()

    doc_ids = args.doc_ids + (pending_doc_ids(args.force) if args.all or not args.doc_ids else [])
    doc_ids = list(dict.fromkeys(doc_ids))
    if not doc_ids:
        print("nothing to parse: every PDF already has parsed JSON (use --all --force to re-parse)")
        return

    JSON_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    converter = build_converter()
    failed = []

    for n, doc_id in enumerate(doc_ids, 1):
        if not (PDF_DIR / f"{doc_id}.pdf").exists():
            print(f"SKIP {doc_id}: {PDF_DIR / doc_id}.pdf not found", file=sys.stderr)
            continue

        print(f"[{n}/{len(doc_ids)}] converting {doc_id} ...", flush=True)
        start = time.monotonic()
        try:
            parse_one(converter, doc_id)
        except Exception as e:
            print(f"FAIL {doc_id}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failed.append(doc_id)
            continue
        print(f"  {time.monotonic() - start:.0f}s", flush=True)

    if failed:
        print(f"done with {len(failed)} failure(s): {' '.join(failed)}")
        sys.exit(1)
    print("done.")


if __name__ == "__main__":
    main()
