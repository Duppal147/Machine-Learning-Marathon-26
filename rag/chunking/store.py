"""JSONL persistence and lookup of chunks by id."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .models import Chunk


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk]):
        self._chunks: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            if c.chunk_id in self._chunks:
                raise ValueError(f"duplicate chunk_id {c.chunk_id}")
            self._chunks[c.chunk_id] = c
            self._by_doc[c.doc_id].append(c.chunk_id)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with open(path) as f:
            return cls(Chunk.from_dict(json.loads(line)) for line in f if line.strip())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for c in self._chunks.values():
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    def get(self, chunk_id: str) -> Chunk:
        return self._chunks[chunk_id]

    def by_doc(self, doc_id: str) -> list[Chunk]:
        return [self._chunks[i] for i in self._by_doc.get(doc_id, [])]

    def neighbors(self, chunk_id: str, k: int = 1) -> list[Chunk]:
        """The chunk plus up to k chunks either side of it in document order (for context expansion)."""
        chunk = self.get(chunk_id)
        ids = self._by_doc[chunk.doc_id]
        i = ids.index(chunk_id)
        return [self._chunks[j] for j in ids[max(0, i - k):i + k + 1]]

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks.values())
