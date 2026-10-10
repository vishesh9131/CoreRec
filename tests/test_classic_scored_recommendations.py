import pytest

from tests.test_model_contract import MODELS, _build


CLASSIC = [m for m in MODELS if m[0] in {'itemknn', 'userknn', 'ease', 'slim', 'als', 'item2vec'}]


@pytest.mark.parametrize('model_id,module_path,cls_name,kwargs', CLASSIC, ids=[m[0] for m in CLASSIC])
def test_classic_scores_and_seen_controls_survive_persistence(model_id, module_path, cls_name, kwargs, tmp_path):
    model = _build(module_path, cls_name, kwargs)
    model.fit(['01', '01', '02', '02', '03', '03'], ['a', 'b', 'b', 'c', 'c', 'd'])
    path = tmp_path / 'model'
    model.save(path)
    for candidate in (model, type(model).load(path)):
        ids = candidate.recommend('01', top_k=4)
        assert set(ids) == {'c', 'd'}
        scored = candidate.recommend('01', top_k=4, return_scores=True)
        assert [item for item, score in scored] == ids
        for item, score in scored:
            assert isinstance(score, float)
            assert score == pytest.approx(candidate.predict('01', item))
        assert set(candidate.recommend('01', top_k=4, exclude_seen=False)) == {'a', 'b', 'c', 'd'}
        assert set(candidate.recommend('01', top_k=4, exclude_seen=False, exclude_items=['d'])) == {'a', 'b', 'c'}
        assert candidate.recommend('unknown', return_scores=True) == []
        assert candidate.recommend('01', top_k=0, return_scores=True) == []
