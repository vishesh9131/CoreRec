import pandas as pd
import pytest

from corerec.api.dataset import RecommenderDataset
from tests.test_model_contract import MODELS, _build


@pytest.mark.parametrize('model_id,module_path,cls_name,kwargs', MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.parametrize('form', ['lists', 'frame', 'dataset'])
def test_mixed_ids_keep_distinct_histories_after_load(model_id, module_path, cls_name, kwargs, form, tmp_path):
    if model_id == 'sasrec':
        kwargs = {**kwargs, 'device': 'cpu'}
    model = _build(module_path, cls_name, kwargs)
    users = [1, 1, '1', '1', 2, 2]
    items = [10, '10', '10', 20, 20, 30]
    ratings = [1.] * 6
    if form == 'lists':
        model.fit(users, items, ratings)
    elif form == 'frame':
        model.fit(pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings}))
    else:
        model.fit(RecommenderDataset.from_triplet(users, items, ratings))
    path = tmp_path / 'model'
    model.save(path)
    for candidate in (model, type(model).load(path)):
        assert set(candidate.recommend(1, top_k=4)) == {20, 30}
        assert set(candidate.recommend('1', top_k=4)) == {10, 30}
        assert set(candidate.recommend(2, top_k=4)) == {10, '10'}
        assert set(candidate.recommend(1, top_k=4, exclude_seen=False)) == {10, '10', 20, 30}
