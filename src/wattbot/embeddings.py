"""Text embeddings through the OpenAI-compatible BadgerBrain gateway."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from . import config

BATCH_SIZE = 16


class EmbeddingConfigurationError(RuntimeError):
    """Raised when the embedding client cannot be configured safely."""


class EmbeddingValidationError(ValueError):
    """Raised when an embedding response violates the model contract."""


def create_embedding_client() -> Any:
    """Create an OpenAI-compatible client from environment-backed settings."""
    if not config.API_KEY:
        raise EmbeddingConfigurationError(
            "OPENAI_API_KEY must be set in the process environment"
        )

    # Imported lazily so validation and mock tests do not require network tooling.
    from openai import OpenAI

    return OpenAI(
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        timeout=config.REQUEST_TIMEOUT,
    )


def _coerce_finite_vector(vector: Sequence[float], *, label: str) -> list[float]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise EmbeddingValidationError(f"{label} is not a numeric vector")

    values = []
    for position, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingValidationError(
                f"{label} contains a nonnumeric value at position {position}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingValidationError(
                f"{label} contains a nonfinite value at position {position}"
            )
        values.append(number)
    return values


def embedding_l2_norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm of a finite numeric vector."""
    values = _coerce_finite_vector(vector, label="vector")
    return math.sqrt(math.fsum(value * value for value in values))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity without assuming inputs are normalized."""
    left_values = _coerce_finite_vector(left, label="left vector")
    right_values = _coerce_finite_vector(right, label="right vector")
    if len(left_values) != len(right_values):
        raise EmbeddingValidationError("vectors must have the same dimension")

    left_norm = embedding_l2_norm(left_values)
    right_norm = embedding_l2_norm(right_values)
    if left_norm == 0.0 or right_norm == 0.0:
        raise EmbeddingValidationError("cosine similarity is undefined for a zero vector")

    dot_product = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    )
    return dot_product / (left_norm * right_norm)


def _validate_embedding(vector: Sequence[float], *, index: int) -> list[float]:
    values = _coerce_finite_vector(vector, label=f"embedding {index}")
    if len(values) != config.EMBEDDING_DIM:
        raise EmbeddingValidationError(
            f"embedding {index} has dimension {len(values)}; "
            f"expected {config.EMBEDDING_DIM}"
        )
    if embedding_l2_norm(values) == 0.0:
        raise EmbeddingValidationError(f"embedding {index} is a zero vector")
    return values


def _ordered_response_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None:
        raise EmbeddingValidationError("embedding response is missing data")
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        raise EmbeddingValidationError("embedding response data is not a sequence")
    if len(data) != expected_count:
        raise EmbeddingValidationError(
            f"embedding response returned {len(data)} vectors; expected {expected_count}"
        )

    embeddings_by_index = {}
    for fallback_index, item in enumerate(data):
        index = getattr(item, "index", fallback_index)
        if isinstance(index, bool) or not isinstance(index, int):
            raise EmbeddingValidationError("embedding response contains an invalid index")
        if index < 0 or index >= expected_count or index in embeddings_by_index:
            raise EmbeddingValidationError(
                f"embedding response contains an invalid or duplicate index: {index}"
            )
        vector = getattr(item, "embedding", None)
        embeddings_by_index[index] = _validate_embedding(vector, index=index)

    if set(embeddings_by_index) != set(range(expected_count)):
        raise EmbeddingValidationError("embedding response indices are incomplete")
    return [embeddings_by_index[index] for index in range(expected_count)]


def embed_texts(texts: list[str], client: Any | None = None) -> list[list[float]]:
    """Embed text in batches while preserving input order."""
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"text at index {index} is not a string")
        if not text.strip():
            raise ValueError(f"text at index {index} is empty")
    if not texts:
        return []

    if client is None:
        client = create_embedding_client()

    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend(_ordered_response_embeddings(response, len(batch)))
    return embeddings


def embed_text(text: str, client: Any | None = None) -> list[float]:
    """Embed one arbitrary text string as a 4096-dimensional vector."""
    return embed_texts([text], client=client)[0]
