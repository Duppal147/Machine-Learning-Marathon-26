"""Format-agnostic data model shared by every stage of the chunking pipeline.

Loaders produce a Document of Blocks; chunkers turn Blocks into Chunks. Nothing
downstream of a loader should know which file format the document came from.
"""
from dataclasses import asdict, dataclass, field
from typing import Literal

BlockKind = Literal["text", "list_item", "table", "caption", "formula", "code"]


@dataclass
class Block:
    """One atomic piece of document content in reading order."""

    doc_id: str
    kind: BlockKind
    text: str
    heading_path: tuple[str, ...]  # section headings above this block, outermost first (excludes doc title)
    page: int | None = None
    source_refs: list[str] = field(default_factory=list)  # e.g. Docling self_refs like "#/texts/42"


@dataclass
class Document:
    doc_id: str
    title: str
    blocks: list[Block]
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit.

    `text` is what gets shown to the LLM and cited; `embed_text` is what gets embedded.
    Enrichers only modify `embed_text`, so recall strategies can be swapped without re-chunking.
    """

    doc_id: str
    text: str
    heading_path: tuple[str, ...]
    pages: list[int]
    kind: str  # "text" or "table"
    source_refs: list[str]
    embed_text: str = ""
    chunk_id: str = ""  # assigned by the pipeline once the final chunk order is known
    strategy: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.embed_text:
            self.embed_text = self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["heading_path"] = list(self.heading_path)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**{**d, "heading_path": tuple(d["heading_path"])})
