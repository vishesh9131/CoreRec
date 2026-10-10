import pytest
from corerec.engines import DeepFM


@pytest.mark.parametrize('task', ['implicit', 'rating'])
def test_partial_side_features_use_fixed_fields_everywhere(task, tmp_path):
    model = DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, seed=7,
                   task=task, verbose=False, device='cpu')
    model.fit(['u', 'u', 'v', 'v'], ['a', 'b', 'b', 'c'], [1., 2., 3., 4.],
              user_features={'u': {'group': 'a'}}, item_features={'a': {'kind': 'a'}})
    path = tmp_path / 'model'
    model.save(path)
    for candidate in (model, DeepFM.load(path)):
        for user in ('u', 'v'):
            ranked = candidate.recommend(user, top_k=3, exclude_seen=False, return_scores=True)
            assert {item for item, _ in ranked} == {'a', 'b', 'c'}
            assert [score for _, score in ranked] == pytest.approx(
                [candidate.predict(user, item) for item, _ in ranked], abs=1e-6)
            assert len(candidate._prediction_features(user, 'c')) == 4


def test_refit_replaces_catalog_and_side_feature_schema(tmp_path):
    model = DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, seed=7,
                   verbose=False, device='cpu')
    model.fit(['u', 'u', 'v', 'v'], ['a', 'b', 'b', 'c'], [1.] * 4,
              user_features={'u': {'group': 'a'}}, item_features={'a': {'kind': 'a'}})
    model.fit(['x', 'x', 'y', 'y'], ['d', 'e', 'e', 'f'], [1.] * 4)
    assert model.field_dims == [2, 3]
    assert model.user_feature_types == model.item_feature_types == []
    assert set(model.feature_map) == {'user', 'item'}
    assert model.recommend('u') == []
    assert set(model.recommend('x', top_k=3, exclude_seen=False)) == {'d', 'e', 'f'}
    path = tmp_path / 'model'
    model.save(path)
    loaded = DeepFM.load(path)
    assert loaded.batch_predict([('x', 'd'), ('y', 'e')]) == pytest.approx(
        model.batch_predict([('x', 'd'), ('y', 'e')]), abs=1e-6)
