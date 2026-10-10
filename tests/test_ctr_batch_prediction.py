import pytest
import torch

from corerec.api.exceptions import ModelNotFittedError
from corerec.engines import DCN, DeepFM


@pytest.mark.parametrize('cls', [DCN, DeepFM])
@pytest.mark.parametrize('task', ['implicit', 'rating'])
def test_batch_prediction_preserves_scores_and_bounds_forward_calls(cls, task, tmp_path):
    model = cls(embedding_dim=4, epochs=1, batch_size=8, task=task,
                verbose=False, device='cpu', seed=7)
    model.fit(['u', 'u', 'v', 'v'], ['a', 'b', 'b', 'c'], [1., 2., 3., 4.],
              user_features={'u': {'group': 'a'}, 'v': {'group': 'b'}},
              item_features={'a': {'kind': 'a'}, 'b': {'kind': 'b'}, 'c': {'kind': 'a'}})
    pairs = [('u', 'a'), ('v', 'b'), ('u', 'c'), ('v', 'a')] * 8
    pairs[5] = ('missing', 'a')
    pairs[20] = ('u', 'missing')
    model.model.eval()
    with torch.no_grad():
        expected = [0. if model._prediction_features(u, i) is None else
                    model.model(torch.tensor([model._prediction_features(u, i)])).item()
                    for u, i in pairs]
    path = tmp_path / 'bundle'
    model.save(path)
    for candidate in (model, cls.load(path)):
        calls = []
        hook = candidate.model.register_forward_hook(lambda module, args, output: calls.append(len(args[0])))
        try:
            assert candidate.batch_predict(iter(pairs)) == pytest.approx(expected, abs=1e-6)
            assert calls == [7, 8, 7, 8]
            calls.clear()
            assert candidate.batch_predict([]) == []
            assert candidate.batch_predict([('missing', 'a')]) == [0.]
            assert calls == []
            assert candidate.predict(*pairs[0]) == pytest.approx(expected[0], abs=1e-6)
        finally:
            hook.remove()


@pytest.mark.parametrize('cls', [DCN, DeepFM])
def test_unfitted_batch_prediction_raises(cls):
    with pytest.raises(ModelNotFittedError):
        cls().batch_predict([])
