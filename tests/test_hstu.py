"""HSTU, the generative sequential model: it must learn order, not just co-occurrence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from corerec.engines import HSTU  # noqa: E402


def _cycles(n_users=60, n_items=40, length=25, seed=0):
    """Each user walks item k -> k+1 from a random start: only order predicts the next item."""
    rng = np.random.default_rng(seed)
    users, items, times = [], [], []
    for u in range(n_users):
        start = int(rng.integers(0, n_items))
        for k in range(length):
            users.append(f"u{u}")
            items.append(f"i{(start + k) % n_items}")
            times.append(1_000_000 * u + 60 * k)
    return users, items, times


def _next_item_hit_rate(model, users, items, n_items=40, k=5):
    last = {}
    for u, it in zip(users, items):
        last[u] = int(it[1:])
    hits = [f"i{(i + 1) % n_items}" in model.recommend(u, top_k=k, exclude_seen=False) for u, i in last.items()]
    return float(np.mean(hits))


@pytest.mark.parametrize("encoder", ["hstu", "sasrec"])
def test_learns_the_next_item_from_order(encoder):
    users, items, times = _cycles()
    model = HSTU(embedding_dim=32, epochs=30, num_negatives=16, batch_size=16, encoder=encoder)
    model.fit(users, items, timestamps=times)
    assert model.history[-1] < model.history[0] / 5
    assert _next_item_hit_rate(model, users, items) >= 0.9


def test_timestamps_decide_the_order_not_the_row_order():
    users, items, times = _cycles(n_users=10)
    perm = np.random.default_rng(1).permutation(len(users))
    shuffled = HSTU(epochs=1, embedding_dim=16).fit(
        [users[i] for i in perm], [items[i] for i in perm], timestamps=[times[i] for i in perm])
    ordered = HSTU(epochs=1, embedding_dim=16).fit(users, items, timestamps=times)
    for u in ordered.user_sequences:
        got = [shuffled.index_to_item[i] for i in shuffled.user_sequences[u]]
        want = [ordered.index_to_item[i] for i in ordered.user_sequences[u]]
        assert got == want


def test_time_bias_only_when_timestamps_are_given():
    users, items, times = _cycles(n_users=8)
    assert HSTU(epochs=1, embedding_dim=16).fit(users, items, timestamps=times).has_time
    assert not HSTU(epochs=1, embedding_dim=16).fit(users, items).has_time
    assert not HSTU(epochs=1, embedding_dim=16, use_time=False).fit(users, items, timestamps=times).has_time


def test_save_load_keeps_scores_and_id_types(tmp_path):
    users, items, times = _cycles(n_users=12)
    int_items = [int(i[1:]) for i in items]
    model = HSTU(epochs=2, embedding_dim=16).fit(users, int_items, timestamps=times)
    path = tmp_path / "hstu.pkl"
    model.save(str(path))
    loaded = HSTU.load(str(path))
    assert loaded.recommend("u3", top_k=5) == model.recommend("u3", top_k=5)
    assert all(isinstance(i, int) for i in loaded.recommend("u3", top_k=5))
    np.testing.assert_allclose(loaded.score_users(["u3", "u7"]), model.score_users(["u3", "u7"]), atol=1e-6)


def test_recommend_filters():
    users, items, times = _cycles(n_users=10)
    model = HSTU(epochs=2, embedding_dim=16).fit(users, items, timestamps=times)
    seen = {model.index_to_item[i] for i in model.user_sequences["u0"]}
    recs = model.recommend("u0", top_k=10)
    assert not seen & set(recs)
    banned = recs[:3]
    assert not set(banned) & set(model.recommend("u0", top_k=10, exclude_items=banned))
    assert model.recommend("nobody", top_k=5) == []


def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="divisible"):
        HSTU(embedding_dim=50, num_heads=3)
    with pytest.raises(ValueError, match="encoder"):
        HSTU(encoder="gru")
    with pytest.raises(ValueError, match="two or more"):
        HSTU(epochs=1).fit(["a", "b"], ["x", "y"])
