import numpy as np
import pytest

from corerec.engines import DCN


@pytest.mark.parametrize("safe", [True, False])
def test_dcn_seen_filter_survives_save_load(tmp_path, safe):
    model = DCN(embedding_dim=4, deep_layers=[8], epochs=1, device="cpu", seed=1)
    model.fit(["01", "01", "02", "02"], ["a", "b", "b", "c"], [1.] * 4)
    path = tmp_path / "dcn"
    model.save(path, safe=safe)
    if safe:
        restored = DCN.load(path)
    else:
        with pytest.warns(UserWarning):
            restored = DCN.load(path, allow_pickle=True)
    for candidate in (model, restored):
        candidate._score_all_items = lambda user: np.array([3., 2., 1.])
        assert candidate.recommend("01", top_k=3) == ["c"]
        assert candidate.recommend("01", top_k=3, exclude_seen=False) == ["a", "b", "c"]
        assert candidate.recommend("01", top_k=3, exclude_items=["c"]) == []
        assert candidate.recommend("missing", top_k=3) == []


def test_dcn_refit_replaces_seen_history():
    model = DCN(embedding_dim=4, deep_layers=[8], epochs=1, device="cpu", seed=1)
    model.fit([1, 2], [10, 11], [1., 1.])
    model.fit([1, 2], [11, 10], [1., 1.])
    assert model._user_item_interactions == {1: {11}, 2: {10}}
