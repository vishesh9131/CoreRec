"""Every production model exposes users_index / items_index after fit and load (#76).

Before, only some models built an IdIndex (TwoTower, LightGCN, the KNNs,
EASE, SLIM, SAR); ALS, Item2Vec, the VAEs, DCN, DeepFM, SASRec and HSTU kept
only their own maps. The index must name exactly the trained ids and agree
with the codes the model itself uses.
"""

import inspect

import numpy as np
import pytest

import corerec.engines as engines

NAMES = sorted(n for n in engines.MODELS if n != "TFIDFRecommender")  # content model: no users

rng = np.random.default_rng(0)
_pairs = sorted({(f"u{u}", int(i)) for u, i in zip(rng.integers(0, 30, 500), rng.integers(0, 40, 500))})
USERS, ITEMS = [u for u, _ in _pairs], [i for _, i in _pairs]
RATINGS = rng.integers(1, 6, len(_pairs)).astype(float).tolist()


def _model(name):
    cls = getattr(engines, name)
    params = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in (("epochs", 1), ("iterations", 2), ("device", "cpu"), ("verbose", False))
          if k in params}
    return cls(**kw).fit(USERS, ITEMS, RATINGS)


def _check(m):
    assert set(m.users_index.ids) <= set(USERS) and set(m.items_index.ids) == set(ITEMS)
    own = getattr(m, "item_map", None) or getattr(m, "item_to_index", None)
    if isinstance(own, dict) and own:
        assert all(m.items_index.code(i) == c for i, c in own.items())


@pytest.mark.parametrize("name", NAMES)
def test_fit_and_load_expose_the_id_index(name, tmp_path):
    m = _model(name)
    _check(m)
    m.save(tmp_path / "m")
    back = type(m).load(tmp_path / "m")
    _check(back)
    assert back.items_index.as_dict() == m.items_index.as_dict()
    assert back.users_index.ids == m.users_index.ids


def test_unfitted_models_have_no_index():
    assert not hasattr(engines.ALS(), "users_index")
