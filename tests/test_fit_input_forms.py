"""Every way of passing the same interactions to fit() trains the same model.

Stage 1 of #78: user input reaches a model through up to six coercion layers,
and which ones run depends on the model's fit() signature. This pins the
behaviour every model must keep while those layers are merged into one.
"""

import importlib

import pandas as pd
import pytest

from tests.test_model_contract import MODELS, _interactions

# (model id, input form) -> why it can't match yet; xfail until fixed
_NEEDS_RATINGS = "ratings are required for this model; #74 decides whether they're optional"
DIVERGENT = {
    ("sar", "no_ratings"): _NEEDS_RATINGS,
    ("sar", "frame_no_rating"): _NEEDS_RATINGS,
    # a frame without a rating column works (the adapter fills 1.0); fit(users, items) doesn't
    ("dcn", "no_ratings"): _NEEDS_RATINGS,
    ("deepfm", "no_ratings"): _NEEDS_RATINGS,
}

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
def test_every_input_form_trains_the_same_model(model_id, module_path, cls_name, kwargs, form,
                                                request):
    if (model_id, form) in DIVERGENT:
        # strict: once fixed, the entry must come off DIVERGENT
        request.node.add_marker(pytest.mark.xfail(reason=DIVERGENT[(model_id, form)], strict=True))
    if model_id == "sasrec":
        # GPU gradient accumulation varies even for identical inputs; compare parsing on CPU.
        kwargs = {**kwargs, "device": "cpu"}
    cls = getattr(importlib.import_module(module_path), cls_name)
    users, items, _ = _interactions()
    ratings = [1.0] * len(users)  # implicit: "no ratings" must mean all ones

    expected = cls(**kwargs).fit(users, items, ratings).recommend(users[0], top_k=5)
    model = cls(**kwargs)
    _fit(model, form, users, items, ratings)
    assert model.recommend(users[0], top_k=5) == expected
