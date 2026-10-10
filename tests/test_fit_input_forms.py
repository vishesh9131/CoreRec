"""Every way of passing the same interactions to fit() trains the same model.

Stage 1 of #78: user input reaches a model through up to six coercion layers,
and which ones run depends on the model's fit() signature. This pins the
behaviour every model must keep while those layers are merged into one.
"""

import importlib

import pandas as pd
import pytest

from tests.test_model_contract import MODELS, _interactions

REQUIRES_RATINGS = {"sar", "dcn", "deepfm"}

FORMS = ["kwargs", "no_ratings", "frame", "frame_no_rating"]


def _fit(model, form, users, items, ratings):
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})
    if form == "kwargs":
        return model.fit(user_ids=users, item_ids=items, ratings=ratings)
    if form == "no_ratings":
        return model.fit(users, items)
    if form == "frame":
        return model.fit(df)
    return model.fit(df[["user_id", "item_id"]])


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs", MODELS, ids=[m[0] for m in MODELS])
def test_every_input_form_trains_the_same_model(model_id, module_path, cls_name, kwargs, form):
    if model_id == "sasrec":
        # GPU gradient accumulation varies even for identical inputs; compare parsing on CPU.
        kwargs = {**kwargs, "device": "cpu"}
    cls = getattr(importlib.import_module(module_path), cls_name)
    users, items, _ = _interactions()
    ratings = [1.0] * len(users)  # explicit positives for models that require ratings

    expected = cls(**kwargs).fit(users, items, ratings).recommend(users[0], top_k=5)
    model = cls(**kwargs)
    if model_id in REQUIRES_RATINGS and form in {"no_ratings", "frame_no_rating"}:
        from corerec.api.exceptions import InvalidDataError
        with pytest.raises(InvalidDataError, match="Explicit ratings are required"):
            _fit(model, form, users, items, ratings)
        assert not model.is_fitted
        return
    _fit(model, form, users, items, ratings)
    assert model.recommend(users[0], top_k=5) == expected


@pytest.mark.parametrize("model_id,module_path,cls_name,kwargs",
                         [m for m in MODELS if m[0] in REQUIRES_RATINGS],
                         ids=[m[0] for m in MODELS if m[0] in REQUIRES_RATINGS])
@pytest.mark.parametrize("form", ["triplet_dataset", "frame_dataset", "kwargs"])
def test_missing_dataset_ratings_preserves_fitted_model(model_id, module_path, cls_name, kwargs, form):
    from corerec.api.dataset import RecommenderDataset
    from corerec.api.exceptions import InvalidDataError

    cls = getattr(importlib.import_module(module_path), cls_name)
    users, items, ratings = _interactions()
    model = cls(**kwargs).fit(users, items, ratings)
    before = model.recommend(users[0], top_k=5)
    with pytest.raises(InvalidDataError, match="Explicit ratings are required"):
        if form == "triplet_dataset":
            model.fit(RecommenderDataset.from_triplet(users, items))
        elif form == "frame_dataset":
            model.fit(RecommenderDataset.from_dataframe(
                pd.DataFrame({"user_id": users, "item_id": items})))
        else:
            model.fit(user_ids=users, item_ids=items, ratings=None)
    assert model.is_fitted
    assert model.recommend(users[0], top_k=5) == before
