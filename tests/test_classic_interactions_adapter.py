import numpy as np
import pandas as pd
import pytest

from corerec.api.dataset import RecommenderDataset
from corerec.api.exceptions import InvalidDataError
from corerec.engines import ItemKNN, UserKNN, EASE, SLIM


@pytest.mark.parametrize("cls", [ItemKNN, UserKNN, EASE, SLIM])
def test_all_forms_use_native_adapter_and_preserve_predictions(cls, tmp_path, monkeypatch):
    users, items, ratings = [2, 1, 2, 3], [30, 10, 20, 10], [1., 2., 3., 1.]
    df = pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings})
    def old_path(*args, **kwargs):
        raise AssertionError("native fit must not call the old unpacker")
    monkeypatch.setattr(cls, "_unpack_fit_args", old_path)
    models = [cls().fit(users, items, ratings), cls().fit(df),
              cls().fit(RecommenderDataset.from_dataframe(df)),
              cls().fit(user_ids=users, item_ids=items, ratings=ratings)]
    for model in models:
        assert model.user_map == model.users_index.as_dict() == {1: 0, 2: 1, 3: 2}
        assert model.item_map == model.items_index.as_dict() == {10: 0, 20: 1, 30: 2}
        np.testing.assert_array_equal(model.R.toarray(), models[0].R.toarray())
        assert model.recommend(1, top_k=3) == models[0].recommend(1, top_k=3)
    models[0].save(tmp_path / "m")
    restored = cls.load(tmp_path / "m")
    assert restored.items_index.as_dict() == models[0].item_map


def test_invalid_input_does_not_replace_fitted_state():
    model = ItemKNN().fit([1, 2], [10, 20])
    original = model.R
    with pytest.raises(InvalidDataError):
        model.fit([1, 2], [10], [1, 2])
    assert model.R is original


def test_mixed_and_numeric_string_ids_remain_distinct():
    model = ItemKNN().fit([1, "1", "01"], [10, "10", "010"])
    assert len(model.user_map) == len(model.item_map) == 3
    assert set(model.user_map) == {1, "1", "01"}
