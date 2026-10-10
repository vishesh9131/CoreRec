import pytest
from tests.test_model_contract import MODELS, _build


@pytest.mark.parametrize('model_id,module,cls,kwargs', MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.parametrize('loaded', [False, True], ids=['fresh', 'loaded'])
def test_refit_replaces_catalog_and_history(model_id, module, cls, kwargs, loaded, tmp_path):
    if model_id == 'sasrec':
        kwargs = {**kwargs, 'device': 'cpu'}
    model = _build(module, cls, kwargs)
    model.fit([0, 0, 1, 1], [10, 11, 11, 12], [1.] * 4)
    if loaded:
        initial = tmp_path / 'initial'
        model.save(initial)
        model = type(model).load(initial)
    model.fit([2, 2, 3, 3], [20, 21, 21, 22], [1.] * 4)
    final = tmp_path / 'refit'
    model.save(final)
    for candidate in (model, type(model).load(final)):
        assert candidate.recommend(0, top_k=3) == []
        assert set(candidate.recommend(2, top_k=3, exclude_seen=False)) == {20, 21, 22}
        assert candidate.recommend(2, top_k=3) == [22]
        assert candidate.recommend(3, top_k=3) == [20]


def test_sar_first_fit_preserves_preconfigured_index_order():
    import pandas as pd
    from corerec.engines import SAR

    frame = pd.DataFrame({'user_id': [9, 7], 'item_id': ['b', 'a'], 'rating': [1., 1.]})
    model = SAR(col_user='user_id', col_item='item_id')
    model._set_index(frame)
    expected_users = dict(model.user2index)
    expected_items = dict(model.item2index)
    model.fit(frame.iloc[::-1])
    assert model.user2index == expected_users
    assert model.item2index == expected_items


def test_sar_failed_refit_cannot_score_stale_matrices(monkeypatch):
    from corerec.engines import SAR
    from corerec.api.exceptions import ModelNotFittedError

    model = SAR()
    model.fit([0, 0, 1, 1], [10, 11, 11, 12], [1.] * 4)
    def fail(*args, **kwargs):
        raise RuntimeError('training failed')
    monkeypatch.setattr(model, '_compute_affinity_matrix', fail)
    with pytest.raises(RuntimeError, match='training failed'):
        model.fit([2, 2, 3, 3], [20, 21, 21, 22], [1.] * 4)
    with pytest.raises(ModelNotFittedError):
        model.recommend(2)
