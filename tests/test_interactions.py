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
