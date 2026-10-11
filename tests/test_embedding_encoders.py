"""corerec.embeddings text and multimodal encoders, with stub models (#79)."""

import numpy as np
import pytest

from corerec.embeddings import MultimodalEncoder, TextEncoder


class StubSentenceModel:
    """Stands in for a SentenceTransformer: fixed vectors per text."""

    VECS = {"red shoes": [3.0, 4.0], "blue shoes": [4.0, 3.0], "hat": [0.0, 2.0]}

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        return np.array([self.VECS[t] for t in texts], dtype=np.float32)


def test_text_encoder_normalizes_and_handles_single_and_batch():
    enc = TextEncoder(StubSentenceModel())
    one = enc.encode("red shoes")
    assert one.shape == (2,) and np.isclose(np.linalg.norm(one), 1.0)
    assert enc.encode(["red shoes", "hat"]).shape == (2, 2)
    assert enc.embedding_dim == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_similarity_is_a_cosine_either_way(normalize):
    """With normalize=False it returned the raw dot product (24.0, not 0.96)."""
    enc = TextEncoder(StubSentenceModel(), normalize=normalize)
    assert enc.similarity("red shoes", "blue shoes") == pytest.approx(24 / 25)
    assert enc.similarity(np.array([3.0, 4.0]), np.array([6.0, 8.0])) == pytest.approx(1.0)


class Fixed:
    def __init__(self, dim, value):
        self.embedding_dim, self.value = dim, value

    def encode(self, data):
        return np.full(self.embedding_dim, self.value, dtype=np.float32)


def test_concat_keeps_one_layout_when_a_modality_is_missing():
    """A missing modality used to shorten the vector, so batches couldn't stack."""
    enc = MultimodalEncoder({"text": Fixed(2, 1.0), "image": Fixed(3, 2.0)})
    full = enc.encode({"text": "t", "image": "i"})
    partial = enc.encode({"text": "t"})
    assert full.tolist() == [2, 2, 2, 1, 1]          # sorted: image, text
    assert partial.tolist() == [0, 0, 0, 1, 1]       # image slot zero-filled
    assert enc.encode_batch([{"text": "t", "image": "i"}, {"text": "t"}]).shape == (2, 5)
    assert enc.embedding_dim == 5


def test_concat_without_a_known_dim_says_why():
    enc = MultimodalEncoder({"text": Fixed(2, 1.0), "image": lambda x: np.ones(3)})
    with pytest.raises(ValueError, match="'image' is missing and its encoder has no embedding_dim"):
        enc.encode({"text": "t"})


def test_average_weighted_and_strict_missing():
    enc = MultimodalEncoder({"a": Fixed(2, 1.0), "b": Fixed(2, 3.0)}, fusion="average")
    assert enc.encode({"a": 0, "b": 0}).tolist() == [2.0, 2.0]
    w = MultimodalEncoder({"a": Fixed(2, 1.0)}, fusion="weighted").add_encoder("b", Fixed(2, 3.0), weight=3.0)
    assert w.encode({"a": 0, "b": 0}).tolist() == [2.5, 2.5]
    with pytest.raises(ValueError, match="Missing modality: b"):
        enc.encode({"a": 0}, missing_ok=False)
    with pytest.raises(ValueError, match="different dims"):
        MultimodalEncoder({"a": Fixed(2, 1.0), "b": Fixed(3, 1.0)}, fusion="average").encode({"a": 0, "b": 0})
    with pytest.raises(ValueError, match="No modalities"):
        enc.encode({})
