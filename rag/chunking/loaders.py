"""Loaders: the only code that knows about a specific input format.

Each loader turns one source file into a format-agnostic Document of Blocks.
To support a new format (e.g. structured markdown), add a class with a
`load(path) -> Document` method; nothing downstream needs to change.
"""
import json
import re
from pathlib import Path
from typing import Protocol

from .models import Block, Document


class DocumentLoader(Protocol):
    def load(self, path: Path) -> Document: ...


# ---------------------------------------------------------------------------
# Heading hierarchy inference
#
# Docling reports every PDF section_header as level 1, so nesting has to be
# recovered from the heading text itself ("III." > "A.", "3" > "3.1" > "3.1.2").
# ---------------------------------------------------------------------------

_DOTTED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")  # 1-2 digits, so "2023 Year in Review" isn't numbering
_ROMAN = re.compile(r"^([IVXLC]+)\.\s+\S")
_LETTER = re.compile(r"^([A-Z])\.\s+\S")
_APPENDIX = re.compile(r"^appendix\b", re.IGNORECASE)

# Unnumbered headings that always start a new top-level section.
TOP_LEVEL_NAMES = {
    "abstract", "references", "bibliography",
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "appendix", "appendices", "executive summary", "table of contents", "contents",
    "list of figures", "list of tables", "citations and bibliography",
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s: str) -> int:
    total = 0
    for ch, nxt in zip(s, s[1:] + " "):
        v = _ROMAN_VALUES[ch]
        total += -v if _ROMAN_VALUES.get(nxt, 0) > v else v
    return total


def normalize_heading(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


class HeadingTracker:
    """Maintains the current heading path while walking a document in reading order."""

    def __init__(self):
        self.stack: list[tuple[int, str]] = []
        self.last_roman = 0
        self.numbered_level = 0  # level of the most recent numbered heading; unnumbered ones nest under it

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.stack)

    def infer_level(self, text: str) -> int:
        """Return the nesting level (1 = top) implied by a heading's numbering."""
        if m := _ROMAN.match(text):
            value = _roman_to_int(m.group(1))
            # "C." / "V." are ambiguous: roman only if they continue the roman sequence.
            if len(m.group(1)) > 1 or value in (1, self.last_roman + 1):
                self.last_roman = value
                self.numbered_level = 1
                return 1
        if _LETTER.match(text):
            level = 2 if self.last_roman else 1
            self.numbered_level = level
            return level
        if m := _DOTTED.match(text):
            level = m.group(1).count(".") + 1
            self.numbered_level = level
            return level
        if _APPENDIX.match(text) or text.lower().strip(" :") in TOP_LEVEL_NAMES:
            self.numbered_level = 0
            return 1
        return self.numbered_level + 1

    def is_numbered(self, text: str) -> bool:
        return bool(_ROMAN.match(text) or _LETTER.match(text) or _DOTTED.match(text))

    def push(self, text: str) -> None:
        text = normalize_heading(text)
        level = self.infer_level(text)
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))


def looks_like_heading(text: str) -> bool:
    """Reject headers Docling mislabels, e.g. infographic stats like "680M" or "#1", or reference fragments."""
    return text[:1].isalnum() and sum(ch.isalpha() for ch in text) >= 3


# ---------------------------------------------------------------------------
# Docling JSON
# ---------------------------------------------------------------------------

_TEXT_KINDS = {
    "text": "text", "paragraph": "text", "footnote": "text",
    "list_item": "list_item", "caption": "caption", "formula": "formula", "code": "code",
}
_SKIP_LABELS = {"page_header", "page_footer"}


def _escape_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def table_to_markdown(grid: list[list[dict]]) -> str:
    rows = [[_escape_cell(cell.get("text", "")) for cell in row] for row in grid]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


class DoclingJsonLoader:
    """Loads the dicts written by `DoclingDocument.export_to_dict()` (see parse_pdfs.py)."""

    def load(self, path: Path) -> Document:
        path = Path(path)
        self._doc = json.loads(path.read_text())
        self._doc_id = path.stem
        self._title: str | None = None
        self._headings = HeadingTracker()
        self._blocks: list[Block] = []

        for ref in self._doc["body"]["children"]:
            self._walk(ref["$ref"])

        return Document(
            doc_id=self._doc_id,
            title=self._title or self._doc.get("name", self._doc_id),
            blocks=self._blocks,
            metadata={"source_path": str(path), "format": "docling_json"},
        )

    def _resolve(self, ref: str) -> tuple[str, dict]:
        _, collection, idx = ref.split("/")  # "#/texts/12"
        return collection, self._doc[collection][int(idx)]

    @staticmethod
    def _page(item: dict) -> int | None:
        prov = item.get("prov") or []
        return prov[0]["page_no"] if prov else None

    def _emit(self, kind: str, text: str, item: dict, refs: list[str] | None = None) -> None:
        text = text.strip()
        if not text:
            return
        self._blocks.append(Block(
            doc_id=self._doc_id,
            kind=kind,
            text=text,
            heading_path=self._headings.path,
            page=self._page(item),
            source_refs=refs or [item["self_ref"]],
        ))

    def _walk(self, ref: str) -> None:
        collection, item = self._resolve(ref)
        if item.get("content_layer", "body") != "body":
            return

        if collection == "texts":
            self._handle_text(item)
            for child in item.get("children", []):
                self._walk(child["$ref"])
        elif collection == "groups":
            for child in item.get("children", []):
                self._walk(child["$ref"])
        elif collection == "tables":
            self._handle_table(item)
        elif collection == "pictures":
            # Text inside a picture is OCR'd chart labels; only the caption is useful.
            for cap in item.get("captions", []):
                _, cap_item = self._resolve(cap["$ref"])
                self._emit("caption", cap_item["text"], cap_item)

    def _handle_text(self, item: dict) -> None:
        label, text = item["label"], item["text"]
        if label in _SKIP_LABELS:
            return
        if label in ("section_header", "title"):
            heading = normalize_heading(text)
            if heading.lower().startswith(("table ", "figure ", "fig. ")):
                self._emit("caption", heading, item)
            elif not looks_like_heading(heading):
                self._emit("text", heading, item)
            elif self._is_title(heading, item):
                self._title = heading
            else:
                self._headings.push(heading)
            return
        kind = _TEXT_KINDS.get(label, "text")
        self._emit(kind, f"- {text}" if kind == "list_item" else text, item)

    def _is_title(self, heading: str, item: dict) -> bool:
        """The first ordinary (unnumbered, not "Abstract") heading on page 1, before any section starts."""
        return (
            self._title is None
            and not self._headings.stack
            and self._page(item) in (1, None)
            and not self._headings.is_numbered(heading)
            and heading.lower() not in TOP_LEVEL_NAMES
        )

    def _handle_table(self, item: dict) -> None:
        parts, refs = [], [item["self_ref"]]
        for cap in item.get("captions", []):
            _, cap_item = self._resolve(cap["$ref"])
            parts.append(cap_item["text"])
            refs.append(cap_item["self_ref"])
        parts.append(table_to_markdown(item.get("data", {}).get("grid", [])))
        for note in item.get("footnotes", []):
            _, note_item = self._resolve(note["$ref"])
            parts.append(note_item["text"])
            refs.append(note_item["self_ref"])
        self._emit("table", "\n\n".join(p for p in parts if p), item, refs)
