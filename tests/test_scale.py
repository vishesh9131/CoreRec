"""Scale regressions: things that were fine at 1k users and impossible at 100k.

Findings/scale/probe.py measures the zoo at larger sizes; these pin the fixes.
"""

import numpy as np
import pandas as pd
import pytest
import torch


def test_two_tower_fits_without_an_identity_matrix():
    """np.eye(n_users) was 160 GB here; ids go straight into the first layer now."""
    from corerec.engines import TwoTower

    n = 200_000
    rng = np.random.default_rng(0)
    users = np.arange(n)
    items = rng.integers(0, 500, n)
    m = TwoTower(embedding_dim=8, hidden_dims=[16], epochs=1, batch_size=8192, verbose=False)
    m.fit(users.tolist(), items.tolist(), [1.0] * n)
    assert len(m.recommend(int(users[0]), top_k=5)) == 5


def test_tower_id_input_matches_one_hot_input():
    from corerec.core.towers import UserTower

    torch.manual_seed(0)
    tower = UserTower(input_dim=50, output_dim=8, config={"hidden_dims": [16], "norm": "batch"}).eval()
    ids = torch.tensor([[3], [17], [49]])
    one_hot = torch.nn.functional.one_hot(ids.squeeze(1), 50).float()
    assert torch.allclose(tower(ids), tower(one_hot), atol=1e-6)


def test_normalize_interactions_is_sparse_and_keeps_last_rating():
    import scipy.sparse as sp

    from corerec.api.base_recommender import normalize_interactions

    users, items, m = normalize_interactions(["b", "a", "b"], [1, 2, 1], [1.0, 2.0, 5.0])
    assert sp.issparse(m)
    assert users == ["b", "a"] and items == [1, 2]
    assert m.toarray().tolist() == [[5.0, 0.0], [0.0, 2.0]]


def test_sasrec_keeps_event_order():
    """Sequences used to come out sorted by item index, not by when things happened."""
    from corerec.engines import SASRec

    users = [0, 0, 0, 1, 1, 1]
    items = [30, 10, 20, 20, 30, 10]
    m = SASRec(hidden_units=8, num_blocks=1, epochs=1, max_seq_length=5, verbose=False)
    m.fit(users, items, [1.0] * 6)
    back = {v: k for k, v in m.item_to_index.items()}
    assert [back[i] for i in m.user_sequences[0]] == [30, 10, 20]
    assert [back[i] for i in m.user_sequences[1]] == [20, 30, 10]


def test_fit_dataframe_orders_by_timestamp():
    from corerec.engines import SASRec

    df = pd.DataFrame({"user_id": [0, 0, 0], "item_id": [10, 20, 30],
                       "rating": 1.0, "timestamp": [3, 1, 2]})
    m = SASRec(hidden_units=8, num_blocks=1, epochs=1, max_seq_length=5, verbose=False)
    m.fit(df)
    back = {v: k for k, v in m.item_to_index.items()}
    assert [back[i] for i in m.user_sequences[0]] == [20, 30, 10]


@pytest.mark.parametrize("name,kw", [
    ("SASRec", dict(hidden_units=8, num_blocks=1, epochs=1, max_seq_length=5, verbose=False)),
    ("LightGCN", dict(epochs=1)),
])
def test_user_who_saw_every_item_does_not_hang_fit(name, kw):
    """Negative sampling looped forever looking for an item the user hadn't seen."""
    import corerec.engines as E

    m = getattr(E, name)(**kw)
    m.fit([0, 0, 0, 1], [1, 2, 3, 1], [1.0] * 4)
    assert m.is_fitted
