"""Persist chunk embeddings in ChromaDB and search them by question.

The chunks JSONL (wattbot.chunking.ChunkStore) stays the source of truth; this
index holds each chunk's vector, its display text and flat citation metadata.
Search hits carry chunk_id, so ChunkStore.get()/neighbors() still apply.

Indexing is incremental: a chunk is (re-)embedded only if its embed_text or the
embedding model changed, and each batch is written as soon as it is embedded,
so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import config
from .chunking import Chunk

EmbedFn = Callable[[list[str]], list[list[float]]]


def collection_name(strategy: str, model: str = config.EMBEDDING_MODEL) -> str:
    """One collection per (chunking strategy, embedding model), e.g. "hybrid__qwen3-vl-embedding-8b"."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", f"{strategy}__{model}")


def embed_hash(chunk: Chunk, model: str) -> str:
    return hashlib.sha256(f"{model}\n{chunk.embed_text}".encode()).hexdigest()


def format_query(question: str, instruction: str | None = config.QUERY_INSTRUCTION) -> str:
    """Qwen3 embedding query format; documents are embedded without an instruction."""
    return f"Instruct: {instruction}\nQuery: {question}" if instruction else question


def chunk_metadata(chunk: Chunk, model: str) -> dict:
    """Flatten a chunk's citation metadata to the scalar values Chroma accepts."""
    meta = {
        "doc_id": chunk.doc_id,
        "kind": chunk.kind,
        "strategy": chunk.strategy,
        "heading_path": " > ".join(chunk.heading_path),
        "pages": ",".join(str(p) for p in chunk.pages),
        "page_start": chunk.pages[0] if chunk.pages else -1,
        "embed_hash": embed_hash(chunk, model),
    }
    for key in ("ref_id", "title", "url", "year", "doc_type"):
        if chunk.metadata.get(key) not in (None, ""):
            meta[key] = str(chunk.metadata[key])
    return meta


def _is_transient(error: Exception) -> bool:
    """Gateway hiccups worth retrying (timeouts, dropped connections, 429/5xx)."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    try:
        import openai
    except ImportError:
        return False
    return isinstance(error, (openai.APITimeoutError, openai.APIConnectionError,
                              openai.RateLimitError, openai.InternalServerError))


def call_with_retry(fn: Callable, *args, attempts: int = 3, base_delay: float = 5.0,
                    sleep: Callable[[float], None] = time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception as error:
            if attempt == attempts or not _is_transient(error):
                raise
            sleep(base_delay * 2 ** (attempt - 1))


@dataclass
class IndexStats:
    embedded: int = 0
    skipped: int = 0
    deleted: int = 0
    batches: int = 0


@dataclass
class SearchHit:
    chunk_id: str
    score: float  # cosine similarity
    text: str
    metadata: dict


class ChunkVectorStore:
    def __init__(self, strategy: str, path: str = config.CHROMA_DIR, model: str = config.EMBEDDING_MODEL,
                 embed_fn: EmbedFn | None = None, client=None):
        import chromadb

        self.strategy = strategy
        self.model = model
        self._embed_fn = embed_fn
        self.client = client or chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
        self.collection = self._open_collection()

    def _open_collection(self):
        return self.client.get_or_create_collection(
            collection_name(self.strategy, self.model),
            embedding_function=None,  # we always supply vectors; never let Chroma embed
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"strategy": self.strategy, "model": self.model},
        )

    def reset(self) -> None:
        """Drop every vector in this collection (a full re-embed follows on the next index())."""
        self.client.delete_collection(self.collection.name)
        self.collection = self._open_collection()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is None:
            from .embeddings import embed_texts  # lazy: needs the gateway client only when used
            self._embed_fn = embed_texts
        return call_with_retry(self._embed_fn, texts)

    def index(self, chunks: Iterable[Chunk], batch_size: int = 64, prune: bool = False,
              progress: Callable[[IndexStats], None] | None = None) -> IndexStats:
        chunks = list(chunks)
        existing = self.collection.get(include=["metadatas"])
        stored_hash = {i: (m or {}).get("embed_hash") for i, m in zip(existing["ids"], existing["metadatas"])}

        stats = IndexStats()
        todo = []
        for chunk in chunks:
            if stored_hash.get(chunk.chunk_id) == embed_hash(chunk, self.model):
                stats.skipped += 1
            else:
                todo.append(chunk)

        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            vectors = self.embed([c.embed_text for c in batch])
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[chunk_metadata(c, self.model) for c in batch],
            )
            stats.embedded += len(batch)
            stats.batches += 1
            if progress:
                progress(stats)

        if prune:
            stale = sorted(set(stored_hash) - {c.chunk_id for c in chunks})
            if stale:
                self.collection.delete(ids=stale)
            stats.deleted = len(stale)
        return stats

    def search(self, question: str, k: int = 10, where: dict | None = None,
               instruction: str | None = config.QUERY_INSTRUCTION) -> list[SearchHit]:
        return self.search_many([question], k, where, instruction)[0]

    def search_many(self, questions: list[str], k: int = 10, where: dict | None = None,
                    instruction: str | None = config.QUERY_INSTRUCTION) -> list[list[SearchHit]]:
        vectors = self.embed([format_query(q, instruction) for q in questions])
        result = self.collection.query(query_embeddings=vectors, n_results=k, where=where,
                                       include=["distances", "documents", "metadatas"])
        return [
            [SearchHit(i, 1.0 - d, doc, meta) for i, d, doc, meta in zip(ids, dists, docs, metas)]
            for ids, dists, docs, metas in zip(result["ids"], result["distances"],
                                               result["documents"], result["metadatas"])
        ]

    def count(self) -> int:
        return self.collection.count()

    def delete_doc(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
