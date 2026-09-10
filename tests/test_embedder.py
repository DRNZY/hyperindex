import numpy as np
import pytest
from unittest.mock import MagicMock
from hyperindex.embedder import Embedder


def test_embedder_vector_shape_and_norm():
    embedder = Embedder.create_mock_or_real()
    embeddings = embedder.embed_texts(["hello world", "function calculate_sum(a, b)"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    # Unit normalized check
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-4)


def test_embedder_empty_input():
    embedder = Embedder.create_mock_or_real(use_mock=True)
    embeddings = embedder.embed_texts([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 384)
    assert embeddings.dtype == np.float32


def test_embedder_mock_flag_and_provider():
    mock_embedder = Embedder.create_mock_or_real(use_mock=True)
    assert mock_embedder.is_mock is True
    assert mock_embedder.provider == "mock"


def test_embedder_preferred_providers():
    providers = Embedder.get_preferred_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0
    assert "CPUExecutionProvider" in providers or "CUDAExecutionProvider" in providers


def test_embedder_mock_deterministic():
    embedder = Embedder.create_mock_or_real(use_mock=True)
    emb1 = embedder.embed_texts(["sample text"])
    emb2 = embedder.embed_texts(["sample text"])
    np.testing.assert_array_almost_equal(emb1, emb2)


def test_embedder_custom_dim():
    embedder = Embedder(session=None, tokenizer=None, dim=128)
    embeddings = embedder.embed_texts(["custom dim"])
    assert embeddings.shape == (1, 128)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, [1.0], atol=1e-4)


def test_embedder_real_session_inference_with_pooling_and_mask():
    # Verify the mean-pooling and unit-normalization math with simulated session & tokenizer
    dim = 4
    texts = ["hello", "world long text"]

    class MockEncoding:
        def __init__(self, ids, mask, type_ids=None):
            self.ids = ids
            self.attention_mask = mask
            self.type_ids = type_ids or [0] * len(ids)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode_batch.return_value = [
        MockEncoding(ids=[101, 7592, 102, 0], mask=[1, 1, 1, 0]),
        MockEncoding(ids=[101, 2088, 2146, 3793], mask=[1, 1, 1, 1]),
    ]

    # Token embeddings for batch of 2, seq_len 4, dim 4
    # Batch 0 has 3 active tokens (index 0, 1, 2), 1 padded token (index 3)
    # Batch 1 has 4 active tokens (index 0, 1, 2, 3)
    token_embeds = np.zeros((2, 4, dim), dtype=np.float32)
    token_embeds[0, 0] = [1.0, 0.0, 0.0, 0.0]
    token_embeds[0, 1] = [0.0, 2.0, 0.0, 0.0]
    token_embeds[0, 2] = [0.0, 0.0, 3.0, 0.0]
    token_embeds[0, 3] = [99.0, 99.0, 99.0, 99.0]  # padding token, must be masked out!

    token_embeds[1, 0] = [1.0, 1.0, 1.0, 1.0]
    token_embeds[1, 1] = [1.0, 1.0, 1.0, 1.0]
    token_embeds[1, 2] = [1.0, 1.0, 1.0, 1.0]
    token_embeds[1, 3] = [1.0, 1.0, 1.0, 1.0]

    mock_session = MagicMock()
    input_ids_meta = MagicMock()
    input_ids_meta.name = "input_ids"
    att_mask_meta = MagicMock()
    att_mask_meta.name = "attention_mask"
    token_type_meta = MagicMock()
    token_type_meta.name = "token_type_ids"
    mock_session.get_inputs.return_value = [input_ids_meta, att_mask_meta, token_type_meta]
    mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    mock_session.run.return_value = [token_embeds]

    embedder = Embedder(session=mock_session, tokenizer=mock_tokenizer, dim=dim)
    assert embedder.is_mock is False
    assert embedder.provider == "CUDAExecutionProvider"

    res = embedder.embed_texts(texts)
    assert res.shape == (2, dim)

    # Check padding token was masked out:
    # Sum of active tokens for batch 0: [1.0, 2.0, 3.0, 0.0] / 3 = [1/3, 2/3, 3/3, 0]
    # Unit normalized:
    expected_unnorm_0 = np.array([1.0 / 3, 2.0 / 3, 1.0, 0.0], dtype=np.float32)
    expected_0 = expected_unnorm_0 / np.linalg.norm(expected_unnorm_0)
    np.testing.assert_allclose(res[0], expected_0, atol=1e-5)

    # Norms must all be 1.0
    norms = np.linalg.norm(res, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Verify session.run received correct feed dict including token_type_ids
    assert mock_session.run.called
    feed_dict = mock_session.run.call_args[0][1]
    assert "input_ids" in feed_dict
    assert "attention_mask" in feed_dict
    assert "token_type_ids" in feed_dict
