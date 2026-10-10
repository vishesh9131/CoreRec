import pytest
from corerec.engines import DCN, DeepFM


@pytest.mark.parametrize('cls', [DCN, DeepFM])
@pytest.mark.parametrize('task', ['implicit', 'rating'])
def test_typed_side_feature_keys_survive_safe_load(cls, task, tmp_path):
    model = cls(embedding_dim=4, epochs=1, seed=7, task=task, verbose=False, device='cpu')
    user_features = {1: {'group': 'a'}, '1': {'group': 'b'}}
    item_features = {10: {'kind': 'a'}, '10': {'kind': 'b'}, 20: {'kind': 'a'}}
    model.fit([1, 1, '1', '1'], [10, '10', '10', 20], [1., 2., 3., 4.],
              user_features=user_features, item_features=item_features)
    pairs = [(1, 10), ('1', '10'), (1, 20), ('1', 10)]
    expected = model.batch_predict(pairs)
    path = tmp_path / 'bundle'
    model.save(path)
    loaded = cls.load(path)
    assert loaded.user_features == user_features
    assert loaded.item_features == item_features
    assert loaded.batch_predict(pairs) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('cls', [DCN, DeepFM])
def test_older_side_feature_dictionary_bundles_still_load(cls, tmp_path):
    from corerec.api.model_bundle import load_bundle, save_bundle

    model = cls(embedding_dim=4, epochs=1, seed=7, verbose=False, device='cpu')
    model.fit(['u', 'u', 'v', 'v'], ['a', 'b', 'b', 'c'], [1.] * 4,
              user_features={'u': {'group': 'a'}, 'v': {'group': 'b'}},
              item_features={'a': {'kind': 'a'}, 'b': {'kind': 'b'}, 'c': {'kind': 'a'}})
    path = tmp_path / 'bundle'
    model.save(path)
    bundle = load_bundle(path)
    for kind in ('user', 'item'):
        bundle['state'][f'{kind}_features'] = dict(bundle['state'].pop(f'{kind}_features_pairs'))
    save_bundle(path, model_class=f'{cls.__module__}.{cls.__name__}', config=bundle['config'],
                state=bundle['state'], state_dict=bundle['state_dict'], arrays=bundle.get('arrays'))
    loaded = cls.load(path)
    assert loaded.predict('u', 'b') == pytest.approx(model.predict('u', 'b'), abs=1e-6)
