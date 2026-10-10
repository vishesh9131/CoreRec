"""A result may be shorter than top_k; masked candidates must never fill it."""
import pytest

from corerec import engines

MODELS = [
    ("ItemKNN", {}), ("UserKNN", {}), ("EASE", {}),
    ("SLIM", {"max_iter": 10}),
    ("ALS", {"iterations": 1, "factors": 3}),
    ("Item2Vec", {"iterations": 1, "factors": 3}),
    ("MultVAE", {"epochs": 1, "hidden_dim": 4, "latent_dim": 2, "device": "cpu"}),
    ("MultiDAE", {"epochs": 1, "hidden_dim": 4, "latent_dim": 2, "device": "cpu"}),
    ("LightGCN", {"epochs": 1, "n_factors": 4}),
    ("SASRec", {"epochs": 1, "hidden_units": 4, "num_blocks": 1,
                "max_seq_length": 3, "batch_size": 4, "device": "cpu"}),
]


@pytest.mark.parametrize("name,params", MODELS, ids=[name for name, _ in MODELS])
def test_exhausted_catalog_never_returns_seen_excluded_or_padding_items(name, params):
    model = getattr(engines, name)(**params).fit([1, 1, 2, 2], [10, 20, 20, 30])
    assert model.recommend(1, top_k=10) == [30]
    assert model.recommend(1, top_k=10, exclude_items=[30]) == []
    model.fit([1, 1, 1, 2, 2], [10, 20, 30, 20, 30])
    assert model.recommend(1, top_k=10) == []


@pytest.mark.parametrize("name,params", MODELS[:8], ids=[name for name, _ in MODELS[:8]])
def test_cf_zero_and_negative_limits(name, params):
    model = getattr(engines, name)(**params).fit([1, 1, 2, 2], [10, 20, 20, 30])
    assert model.recommend(1, top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        model.recommend(1, top_k=-1)
