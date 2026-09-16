"""
Ingest pipeline: download PDFs -> extract text -> chunk -> embed -> store in ChromaDB.

Stages:
  1. Download PDFs from metadata.csv URLs (cached locally)
  2. Extract text with PyMuPDF
  3. Split into word-count chunks with overlap
  4. Embed chunks via BadgerBrain API
  5. Store embeddings + metadata in ChromaDB
"""

import os
import hashlib
import requests
import pandas as pd
import fitz  # PyMuPDF
import chromadb
from openai import OpenAI
from tqdm import tqdm

from . import config


def _get_client():
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=config.REQUEST_TIMEOUT)


def download_pdf(url: str, cache_dir: str) -> str | None:
    os.makedirs(cache_dir, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".pdf"
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        return path
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return path
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return None


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    if client is None:
        client = _get_client()
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings


def get_collection(chroma_dir: str = config.CHROMA_DIR) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name="wattbot_corpus",
        metadata={"hnsw:space": "cosine"},
    )


def ingest(metadata_path: str | None = None, limit: int | None = None):
    if metadata_path is None:
        metadata_path = os.path.join(config.DATA_DIR, "metadata.csv")

    df = pd.read_csv(metadata_path, encoding="utf-8-sig")
    if limit:
        df = df.head(limit)

    collection = get_collection()
    client = _get_client()
    total_chunks = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
        doc_id = row["id"]
        url = row["url"]

        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            continue

        pdf_path = download_pdf(url, config.PDF_CACHE_DIR)
        if pdf_path is None:
            continue

        text = extract_text(pdf_path)
        if not text.strip():
            print(f"  No text extracted from {doc_id}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        embeddings = embed_texts(chunks, client)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": doc_id,
                "title": str(row.get("title", "")),
                "year": str(row.get("year", "")),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"Ingestion complete: {total_chunks} new chunks added.")
    return collection
