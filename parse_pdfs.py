#!/usr/bin/env python3
"""Convert sample PDFs with Docling: structured JSON + extracted figure images.

Usage:
    python parse_pdfs.py doc_id [doc_id ...]

doc_id is the PDF filename without ".pdf", e.g. "2501.16548" or
"2023_Amazon_Sustainability_Report_0186f6c3".
"""
import json
import sys
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


def main():
    doc_ids = sys.argv[1:]
    if not doc_ids:
        print("usage: python parse_pdfs.py doc_id [doc_id ...]", file=sys.stderr)
        sys.exit(1)

    JSON_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    converter = build_converter()

    for doc_id in doc_ids:
        pdf_path = PDF_DIR / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"SKIP {doc_id}: {pdf_path} not found", file=sys.stderr)
            continue

        print(f"converting {doc_id} ...")
        result = converter.convert(str(pdf_path))
        doc = result.document

        json_path = JSON_OUT / f"{doc_id}.json"
        json_path.write_text(json.dumps(doc.export_to_dict(), indent=2))

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

    print("done.")


if __name__ == "__main__":
    main()
