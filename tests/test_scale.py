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


def _cf_data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 150, n).tolist(), rng.integers(0, 120, n).tolist(),
            rng.uniform(0.5, 5, n).tolist())


@pytest.mark.parametrize("name,attr", [("ItemKNN", "S"), ("UserKNN", "Su"), ("SLIM", "W")])
def test_neighbour_matrices_are_sparse(name, attr):
    """These were dense n x n: 40 GB for ItemKNN at 100k items."""
    import scipy.sparse as sp

    import corerec.engines as E

    m = getattr(E, name)(top_k_neighbors=10) if name != "SLIM" else E.SLIM()
    m.fit(*_cf_data())
    M = getattr(m, attr)
    assert sp.issparse(M)
    if name != "SLIM":
        assert np.diff(M.indptr).max() <= 10


def test_old_dense_itemknn_pickle_still_loads(tmp_path):
    from corerec.engines import ItemKNN

    m = ItemKNN(top_k_neighbors=10)
    m.fit(*_cf_data())
    want = m.recommend(0, top_k=10)
    m.S = m.S.toarray()  # what 0.7.0 and earlier pickled
    m.save(str(tmp_path / "old.pkl"))
    assert ItemKNN.load(str(tmp_path / "old.pkl")).recommend(0, top_k=10) == want


def test_ease_refuses_a_catalogue_it_cannot_invert():
    from corerec.engines import EASE

    m = EASE()
    m.num_items = 200_000
    with pytest.raises(MemoryError, match="ItemKNN or ALS"):
        m._fit_model()


def test_vae_never_densifies_all_users(monkeypatch):
    """MultVAE put the whole [n_users, n_items] matrix on the device."""
    import scipy.sparse as sp

    from corerec.engines import MultVAE

    calls = []
    real = sp.csr_matrix.toarray
    monkeypatch.setattr(sp.csr_matrix, "toarray",
                        lambda self, *a, **k: calls.append(self.shape[0]) or real(self, *a, **k))
    u, i, _ = _cf_data()
    MultVAE(epochs=1, batch_size=32).fit(u, i)
    assert max(calls) <= 32


def test_blocked_neighbour_search_matches_one_shot():
    from corerec.engines.classic_cf import _sparse_cosine_topk, csr_matrix

    rng = np.random.default_rng(2)
    X = csr_matrix((rng.uniform(1, 5, 2000).astype(np.float32),
                    (rng.integers(0, 90, 2000), rng.integers(0, 70, 2000))), shape=(90, 70))
    one_shot = _sparse_cosine_topk(X, 0.0, 10)
    blocked = _sparse_cosine_topk(X, 0.0, 10, max_block_entries=70 * 3)
    assert abs(one_shot - blocked).max() < 1e-6


def test_sasrec_predicts_from_the_newest_position():
    """Inference read position len(history)-1 of a left-padded sequence -- a pad
    slot for every short history -- while training reads the last position.
    On this chain task that cost HR@1 0.98 -> 0.75."""
    from corerec.engines import SASRec

    rng = np.random.default_rng(0)
    users, items, nxt = [], [], {}
    for u in range(300):
        s, n = int(rng.integers(0, 40)), int(rng.integers(3, 9))
        for k in range(n):
            users.append(u)
            items.append(s + k)
        nxt[u] = s + n
    torch.manual_seed(0)
    np.random.seed(0)
    m = SASRec(hidden_units=32, num_blocks=1, epochs=15, max_seq_length=20,
               verbose=False, device="cpu")
    m.fit(users, items, [1.0] * len(users))
    hits = [nxt[u] in m.recommend(u, top_k=1) for u in nxt if nxt[u] < 48]
    assert np.mean(hits) > 0.9
