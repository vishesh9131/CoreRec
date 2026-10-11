"""to_interactions: every fit() input form gives the same Interactions (#78)."""

import numpy as np
import pandas as pd
import pytest

from corerec.api.exceptions import InvalidDataError
from corerec.api.interactions import to_interactions

U, I, R, T = ["a", "b", "a", "c"], [10, 20, 30, 10], [1.0, 2.0, 1.0, 3.0], [4, 3, 2, 1]


def _same(x, y):
    assert x.users.ids == y.users.ids and x.items.ids == y.items.ids
    np.testing.assert_array_equal(x.user_codes, y.user_codes)
    np.testing.assert_array_equal(x.item_codes, y.item_codes)
    np.testing.assert_array_equal(x.ratings, y.ratings)


def test_every_form_gives_the_same_interactions():
    ref = to_interactions(U, I, R)
    df = pd.DataFrame({"user_id": U, "item_id": I, "rating": R})
    for got in (to_interactions(user_ids=U, item_ids=I, ratings=R),
                to_interactions(U, I, interactions=R),
                to_interactions(df)):
        _same(got, ref)


def test_missing_ratings_mean_ones_and_offset_reserves_padding():
    x = to_interactions(U, I, item_offset=1)
    assert x.ratings.tolist() == [1.0] * 4
    assert x.item_codes.tolist() == [1, 2, 3, 1]
    assert x.triple() == (U, I, [1.0] * 4)
    _same(to_interactions(pd.DataFrame({"user_id": U, "item_id": I}), item_offset=1), x)


def test_timestamps_by_keyword_or_column():
    a = to_interactions(U, I, timestamps=T)
    b = to_interactions(pd.DataFrame({"user_id": U, "item_id": I, "timestamp": T}))
    assert a.timestamps.tolist() == b.timestamps.tolist() == T


@pytest.mark.parametrize("call,match", [
    (lambda: to_interactions(U, I[:3]), "4 user_ids but 3 item_ids"),
    (lambda: to_interactions(U, I, R[:2]), "2 ratings for 4"),
    (lambda: to_interactions(U, I, [1, float("nan"), 1, 1]), "1 of 4 ratings are NaN"),
    (lambda: to_interactions(U, I, timestamps=T[:1]), "1 timestamps for 4"),
    (lambda: to_interactions(object()), "can't read interactions from object"),
])
def test_bad_input_says_what_is_wrong(call, match):
    with pytest.raises(InvalidDataError, match=match):
        call()


def test_normalize_interactions_reads_the_triple_through_the_adapter():
    """TwoTower and SASRec go through normalize_interactions; its triple branch is now
    to_interactions, so bad input fails as for every other model (#78)."""
    import scipy.sparse as sp

    from corerec.api.base_recommender import normalize_interactions

    users, items, m = normalize_interactions(["b", "a", "b", "b"], [1, 2, 1, 3], [1.0, 2.0, 5.0, 4.0])
    assert users == ["b", "a"] and items == [1, 2, 3]       # first appearance
    assert m.toarray().tolist() == [[5.0, 0, 4.0], [0, 2.0, 0]]  # repeat keeps its last rating
    assert normalize_interactions(["a"], [1], None)[2].toarray().tolist() == [[1.0]]
    dense = np.ones((2, 3), dtype=np.float32)                # matrix form keeps its values
    assert np.array_equal(normalize_interactions(["u", "v"], [1, 2, 3], dense)[2].toarray(), dense)
    assert sp.issparse(normalize_interactions(["u", "v"], [1, 2, 3], sp.csr_matrix(dense))[2])
    with pytest.raises(InvalidDataError, match="1 of 2 ratings are NaN"):
        normalize_interactions(["a", "b"], [1, 2], [1.0, float("nan")])
    with pytest.raises(InvalidDataError, match="3 user_ids but 2 item_ids"):
        normalize_interactions(["a", "b", "c"], [1, 2], None)


def test_the_matrix_form_reads_nonzero_cells_as_events():
    """(unique users, unique items, [n_users, n_items] matrix): the form TwoTower and SASRec take."""
    import scipy.sparse as sp

    m = np.array([[0, 2.5, 0], [0, 0, 0], [1.0, 0, 3.0]])
    for matrix in (m, sp.csr_matrix(m)):
        x = to_interactions(["u1", "u2", "u3"], [10, 20, 30], matrix)
        assert x.users.ids == ["u1", "u2", "u3"]          # u2 has no events but keeps its row
        assert x.items.ids == [10, 20, 30]
        assert x.user_codes.tolist() == [0, 2, 2] and x.item_codes.tolist() == [1, 0, 2]
        assert x.ratings.tolist() == [2.5, 1.0, 3.0]
    padded = to_interactions(["u1", "u2", "u3"], [10, 20, 30], m, item_offset=1)
    assert padded.item_codes.tolist() == [2, 1, 3]


@pytest.mark.parametrize("users,items,matrix,match", [
    (["a", "b"], [1, 2], np.ones((3, 2)), "3x2 interaction matrix needs 3 user_ids"),
    (["a", "a"], [1, 2], np.ones((2, 2)), "each id may appear only once"),
    (["a", "b"], [1, 2], np.array([[1.0, np.nan], [0, 1.0]]), "1 cells of the interaction matrix are NaN"),
])
def test_a_bad_matrix_says_what_is_wrong(users, items, matrix, match):
    with pytest.raises(InvalidDataError, match=match):
        to_interactions(users, items, matrix)


@pytest.mark.parametrize("name,kw", [
    ("TwoTower", {"embedding_dim": 8, "epochs": 2, "verbose": False, "seed": 0}),
    ("SASRec", {"hidden_units": 8, "num_blocks": 1, "num_heads": 1, "epochs": 1, "max_seq_length": 5,
                "verbose": False, "seed": 0}),
])
def test_matrix_and_triple_train_the_same_model(name, kw):
    import corerec.engines as engines

    users, items = ["u1", "u2", "u3"], ["a", "b", "c", "d"]
    m = np.array([[1.0, 0, 1.0, 0], [0, 1.0, 0, 1.0], [1.0, 1.0, 0, 0]])
    rows, cols = m.nonzero()
    cls = getattr(engines, name)
    a = cls(device="cpu", **kw).fit(users, items, m)
    b = cls(device="cpu", **kw).fit([users[r] for r in rows], [items[c] for c in cols], [1.0] * len(rows))
    assert a.recommend("u1", top_k=2) == b.recommend("u1", top_k=2)
