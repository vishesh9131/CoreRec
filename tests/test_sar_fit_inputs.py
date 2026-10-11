"""SAR.fit() reads every input form through to_interactions (#78, stage 2).

Each form must train the same model as the plain DataFrame in SAR's own
column names, and bad input must fail with the adapter's InvalidDataError.
"""

import numpy as np
import pandas as pd
import pytest

from corerec.api.exceptions import InvalidDataError
from corerec.engines import SAR

USERS = [1, 1, 2, 2, 3, 3, 3]
ITEMS = [10, 20, 20, 30, 10, 30, 40]
RATINGS = [5.0, 3.0, 4.0, 2.0, 1.0, 4.5, 3.7]
TS = [100, 200, 300, 400, 500, 600, 700]


def _scores(model):
    return {u: model.recommend(u, top_k=4) for u in (1, 2, 3)}, model.user_affinity.toarray()


def _reference(**kw):
    df = pd.DataFrame({"userID": USERS, "itemID": ITEMS, "rating": RATINGS, "timestamp": TS})
    return _scores(SAR(**kw).fit(df))


@pytest.mark.parametrize("fit", [
    lambda m: m.fit(USERS, ITEMS, RATINGS),
    lambda m: m.fit(USERS, ITEMS, ratings=RATINGS),
    lambda m: m.fit(user_ids=USERS, item_ids=ITEMS, ratings=RATINGS),
    lambda m: m.fit(pd.DataFrame({"user_id": USERS, "item_id": ITEMS, "rating": RATINGS})),
    lambda m: m.fit_from_lists(USERS, ITEMS, RATINGS),
], ids=["positional", "ratings-kw", "keywords", "user_id-frame", "fit_from_lists"])
def test_every_input_form_trains_the_same_model(fit):
    recs, affinity = _scores(fit(SAR()))
    ref_recs, ref_affinity = _reference()
    assert recs == ref_recs
    np.testing.assert_array_equal(affinity, ref_affinity)


def test_a_dataset_trains_the_same_model():
    from corerec.api.dataset import RecommenderDataset

    ds = RecommenderDataset.from_dataframe(
        pd.DataFrame({"user_id": USERS, "item_id": ITEMS, "rating": RATINGS}))
    assert _scores(SAR().fit(ds))[0] == _reference()[0]


def test_timestamps_reach_time_decay_from_the_triple():
    kw = {"timedecay_formula": True, "time_decay_coefficient": 0.01, "time_now": 800}
    recs, affinity = _scores(SAR(**kw).fit(USERS, ITEMS, RATINGS, timestamps=TS))
    ref_recs, ref_affinity = _reference(**kw)
    assert recs == ref_recs
    np.testing.assert_allclose(affinity, ref_affinity)


def test_custom_column_names_still_work():
    df = pd.DataFrame({"who": USERS, "what": ITEMS, "score": RATINGS})
    m = SAR(col_user="who", col_item="what", col_rating="score").fit(df)
    assert m.recommend(1, top_k=4) == _reference()[0][1]


@pytest.mark.parametrize("args,match", [
    ((USERS, ITEMS[:-1], RATINGS), "item_ids"),
    ((USERS, ITEMS, RATINGS[:-1] + [float("nan")]), "NaN"),
    ((USERS, ITEMS, RATINGS[:-1]), "ratings"),
])
def test_bad_input_fails_with_the_adapters_error(args, match):
    """Mismatched lengths used to surface as pandas' 'All arrays must be of the same length'."""
    with pytest.raises(InvalidDataError, match=match):
        SAR().fit(*args)


def test_ratings_are_still_required():
    with pytest.raises(InvalidDataError, match="ratings"):
        SAR().fit(USERS, ITEMS)
