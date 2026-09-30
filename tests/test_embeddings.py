"""Offline embedding contract tests; run with unittest or pytest."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot import config, embeddings


def vector(value=1.0, dimension=4096):
    return [value] + [0.0] * (dimension - 1)


def response(*vectors):
    return SimpleNamespace(data=[
        SimpleNamespace(index=index, embedding=values)
        for index, values in enumerate(vectors)
    ])


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        settings = patch.multiple(
            config, API_KEY="", EMBEDDING_MODEL="qwen3-vl-embedding-8b",
            EMBEDDING_DIM=4096,
        )
        settings.start()
        self.addCleanup(settings.stop)
        # Any unintended attempt to create a real client fails before API access.
        factory = patch.object(
            embeddings, "create_embedding_client",
            side_effect=AssertionError("Tests must not create a live client"),
        )
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock()))
        self.create = self.client.embeddings.create

    def test_embed_text_sends_exact_text_and_qwen_model(self):
        self.create.return_value = response(vector())
        self.assertEqual(
            embeddings.embed_text("  Energy use\n", self.client), vector(),
        )
        self.create.assert_called_once_with(
            model="qwen3-vl-embedding-8b", input=["  Energy use\n"],
        )

    def test_embed_texts_preserves_input_order_for_shuffled_response(self):
        self.create.return_value = response(vector(1), vector(2), vector(3))
        self.create.return_value.data.reverse()
        self.assertEqual(
            embeddings.embed_texts(["first", "second", "third"], self.client),
            [vector(1), vector(2), vector(3)],
        )

    def test_batch_boundaries_and_order_across_shuffled_batches(self):
        for count in (1, 16, 17, 32, 35):
            with self.subTest(count=count):
                self.create.reset_mock()
                texts = [str(index + 1) for index in range(count)]

                def create(*, model, input):
                    result = response(*(vector(int(text)) for text in input))
                    result.data.reverse()
                    return result

                self.create.side_effect = create
                self.assertEqual(
                    embeddings.embed_texts(texts, self.client),
                    [vector(index + 1) for index in range(count)],
                )
                self.assertEqual(self.create.call_args_list, [
                    call(model="qwen3-vl-embedding-8b", input=texts[start:start + 16])
                    for start in range(0, count, 16)
                ])

    def test_empty_list_does_not_create_client_or_request(self):
        self.assertEqual(embeddings.embed_texts([]), [])
        self.assertEqual(embeddings.embed_texts([], self.client), [])
        self.factory.assert_not_called()
        self.create.assert_not_called()

    def test_blank_text_is_rejected_before_requests(self):
        for text in ("", " ", "\n\t"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                embeddings.embed_text(text, self.client)
        self.create.assert_not_called()

    def test_invalid_single_text_is_rejected(self):
        for text in (None, 123, True, b"text", [], {}):
            with self.subTest(text=text), self.assertRaises(TypeError):
                embeddings.embed_text(text, self.client)
        self.create.assert_not_called()

    def test_invalid_container_is_rejected(self):
        for texts in (None, "text", ("text",), {"text"}, {"text": 1}, iter(["text"])):
            with self.subTest(texts=texts), self.assertRaises(TypeError):
                embeddings.embed_texts(texts, self.client)
        self.create.assert_not_called()

    def test_all_inputs_are_validated_before_first_batch(self):
        for invalid, error in ((" ", ValueError), (None, TypeError)):
            with self.subTest(invalid=invalid), self.assertRaises(error):
                embeddings.embed_texts(["valid"] * 16 + [invalid], self.client)
        self.create.assert_not_called()

    def test_default_client_factory_is_used(self):
        self.factory.side_effect = None
        self.factory.return_value = self.client
        self.create.return_value = response(vector())
        self.assertEqual(embeddings.embed_text("text"), vector())
        self.factory.assert_called_once_with()

    def test_model_override_is_sent_to_api(self):
        self.create.return_value = response(vector())
        with patch.object(config, "EMBEDDING_MODEL", "test-model-override"):
            embeddings.embed_text("text", self.client)
        self.create.assert_called_once_with(model="test-model-override", input=["text"])

    def test_missing_or_malformed_response_data(self):
        for result in (None, object(), SimpleNamespace(data=None),
                       SimpleNamespace(data=42), SimpleNamespace(data=True),
                       SimpleNamespace(data="x"), SimpleNamespace(data={0: vector()})):
            with self.subTest(result=result):
                self.create.return_value = result
                with self.assertRaises(embeddings.EmbeddingValidationError):
                    embeddings.embed_text("text", self.client)

    def test_response_count_must_match_input_count(self):
        for result in (response(), response(vector(), vector())):
            with self.subTest(count=len(result.data)):
                self.create.return_value = result
                with self.assertRaisesRegex(embeddings.EmbeddingValidationError, "expected 1"):
                    embeddings.embed_text("text", self.client)

    def test_invalid_response_indices(self):
        for indices in ((0, 0), (-1, 1), (0, 2), (True, 1), (0.0, 1), ("0", 1), (None, 1)):
            with self.subTest(indices=indices):
                self.create.return_value = SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=vector()) for index in indices
                ])
                with self.assertRaisesRegex(embeddings.EmbeddingValidationError, "index"):
                    embeddings.embed_texts(["first", "second"], self.client)

    def test_missing_indices_use_response_order(self):
        self.create.return_value = SimpleNamespace(data=[
            SimpleNamespace(embedding=vector(1)), SimpleNamespace(embedding=vector(2)),
        ])
        self.assertEqual(
            embeddings.embed_texts(["first", "second"], self.client),
            [vector(1), vector(2)],
        )

    def test_missing_embedding_is_rejected(self):
        self.create.return_value = SimpleNamespace(data=[SimpleNamespace(index=0)])
        with self.assertRaises(embeddings.EmbeddingValidationError):
            embeddings.embed_text("text", self.client)

    def test_invalid_vectors_are_rejected(self):
        for values in (None, "vector", b"vector", 42, {0: 1},
                       vector(True), vector("1"), vector(None)):
            with self.subTest(values_type=type(values).__name__):
                self.create.return_value = response(values)
                with self.assertRaises(embeddings.EmbeddingValidationError):
                    embeddings.embed_text("text", self.client)

    def test_wrong_dimension_is_rejected(self):
        for values in ([], vector(dimension=4095), vector(dimension=4097)):
            with self.subTest(dimension=len(values)):
                self.create.return_value = response(values)
                with self.assertRaisesRegex(embeddings.EmbeddingValidationError, "dimension"):
                    embeddings.embed_text("text", self.client)

    def test_nonfinite_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.create.return_value = response(vector(value))
                with self.assertRaisesRegex(embeddings.EmbeddingValidationError, "nonfinite"):
                    embeddings.embed_text("text", self.client)

    def test_zero_vector_is_rejected(self):
        self.create.return_value = response(vector(0))
        with self.assertRaisesRegex(embeddings.EmbeddingValidationError, "zero vector"):
            embeddings.embed_text("text", self.client)

    def test_cosine_similarity_for_unnormalized_vectors(self):
        for left, right, expected in (
            ([3, 0], [6, 0], 1), ([3, 0], [0, 4], 0),
            ([3, 0], [-6, 0], -1), ([3, 4], [4, 3], 0.96),
        ):
            with self.subTest(left=left, right=right):
                self.assertAlmostEqual(embeddings.cosine_similarity(left, right), expected)

    def test_cosine_similarity_rejects_invalid_vectors(self):
        for left, right in (
            ([1], [1, 2]), ([0, 0], [1, 2]), ([1, 2], [0, 0]), ([], []),
            ([float("nan")], [1]), ([1], [float("inf")]), ([True], [1]),
            ([1], ["1"]), (None, [1]),
        ):
            with self.subTest(left=left, right=right):
                with self.assertRaises(embeddings.EmbeddingValidationError):
                    embeddings.cosine_similarity(left, right)

    def test_l2_norm(self):
        self.assertEqual(embeddings.embedding_l2_norm([3, 4]), 5)
        self.assertEqual(embeddings.embedding_l2_norm(vector()), 1)


class ClientConfigurationTests(unittest.TestCase):
    def test_missing_key_fails_without_constructing_client(self):
        sdk = SimpleNamespace(OpenAI=Mock())
        with patch.dict(sys.modules, {"openai": sdk}), patch.object(config, "API_KEY", ""):
            with self.assertRaises(embeddings.EmbeddingConfigurationError):
                embeddings.create_embedding_client()
        sdk.OpenAI.assert_not_called()

    def test_client_uses_configured_gateway_key_and_timeout(self):
        sdk = SimpleNamespace(OpenAI=Mock())
        with patch.dict(sys.modules, {"openai": sdk}), patch.multiple(
            config, BASE_URL="https://gateway.invalid/v1", API_KEY="offline-test-only",
            REQUEST_TIMEOUT=123,
        ):
            self.assertIs(embeddings.create_embedding_client(), sdk.OpenAI.return_value)
        sdk.OpenAI.assert_called_once_with(
            base_url="https://gateway.invalid/v1", api_key="offline-test-only", timeout=123,
        )


if __name__ == "__main__":
    unittest.main()
