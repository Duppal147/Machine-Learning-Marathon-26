"""
Retriever: query expansion -> embed -> vector search -> deduplicate -> rerank.

Stages:
  1. Query expansion: LLM generates alternative search queries
  2. Query embedding: embed original + expanded queries
  3. Vector search: ChromaDB similarity search for each query
  4. Deduplication: merge results across queries
  5. Re-rank: cross-encoder or LLM-based reranking (stub for now)
"""

from openai import OpenAI

from . import config
from .ingest import embed_texts, get_collection


def _get_client():
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=config.REQUEST_TIMEOUT)


def expand_query(question: str, client: OpenAI | None = None, n: int = config.NUM_QUERY_EXPANSIONS) -> list[str]:
    if client is None:
        client = _get_client()

    prompt = (
        f"Generate {n} alternative search queries for finding relevant scientific passages "
        f"to answer this question. Return ONLY the queries, one per line, no numbering.\n\n"
        f"Question: {question}"
    )

    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )

    raw = resp.choices[0].message.content.strip()
    queries = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
    return queries[:n]


def embed_queries(queries: list[str], client: OpenAI | None = None) -> list[list[float]]:
    if client is None:
        client = _get_client()
    return embed_texts(queries, client)


def vector_search(
    query_embeddings: list[list[float]],
    collection=None,
    top_k: int = config.TOP_K_RETRIEVAL,
) -> list[dict]:
    if collection is None:
        collection = get_collection()

    seen_ids = set()
    results = []

    for emb in query_embeddings:
        hits = collection.query(query_embeddings=[emb], n_results=top_k)
        for i, doc_id in enumerate(hits["ids"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                results.append({
                    "chunk_id": doc_id,
                    "text": hits["documents"][0][i],
                    "metadata": hits["metadatas"][0][i],
                    "distance": hits["distances"][0][i] if hits.get("distances") else None,
                })

    return results


def rerank(question: str, chunks: list[dict], client: OpenAI | None = None, top_k: int = config.TOP_K_RERANK) -> list[dict]:
    """Re-rank retrieved chunks by relevance. Uses LLM scoring."""
    if client is None:
        client = _get_client()

    if len(chunks) <= top_k:
        return chunks

    numbered = "\n\n".join(
        f"[{i}] (doc: {c['metadata'].get('doc_id', '?')})\n{c['text'][:500]}"
        for i, c in enumerate(chunks)
    )

    prompt = (
        f"Given this question, rank the following passages by relevance. "
        f"Return ONLY the passage numbers of the top {top_k} most relevant, "
        f"in order from most to least relevant, comma-separated.\n\n"
        f"Question: {question}\n\nPassages:\n{numbered}"
    )

    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        indices = [int(x.strip().strip("[]")) for x in raw.split(",") if x.strip().strip("[]").isdigit()]
        indices = [i for i in indices if 0 <= i < len(chunks)]
        reranked = [chunks[i] for i in indices[:top_k]]
        if reranked:
            return reranked
    except Exception:
        pass

    return chunks[:top_k]


def retrieve(question: str, collection=None, client: OpenAI | None = None) -> list[dict]:
    if client is None:
        client = _get_client()

    alt_queries = expand_query(question, client)
    all_queries = [question] + alt_queries

    query_embs = embed_queries(all_queries, client)
    raw_chunks = vector_search(query_embs, collection)
    ranked_chunks = rerank(question, raw_chunks, client)

    return ranked_chunks
