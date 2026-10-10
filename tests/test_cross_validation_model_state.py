import pandas as pd
import pytest
from corerec.evaluation import CrossValidator


class MemorizingModel:
    def __init__(self):
        self.is_fitted = False
        self.items = set()
        self.fit_calls = 0

    def fit(self, users, items, ratings):
        self.items.update(items)
        self.is_fitted = True
        self.fit_calls += 1
        return self

    def recommend(self, user_id, top_k=10, **kwargs):
        return sorted(self.items)[:top_k]


@pytest.fixture
def interactions():
    return pd.DataFrame({'user_id': [0] * 4, 'item_id': [10, 11, 12, 13], 'rating': [1.] * 4})


@pytest.mark.parametrize('factory', [False, True])
def test_pretrained_models_rejected_before_training(factory, interactions):
    model = MemorizingModel().fit([0] * 4, [10, 11, 12, 13], [1.] * 4)
    if not factory:
        model.__deepcopy__ = lambda memo: pytest.fail('pretrained model must not be copied')
    supplied = (lambda: model) if factory else model
    with pytest.raises(ValueError, match='unfitted model'):
        CrossValidator(2).cross_validate(supplied, interactions, metric='Recall@4')
    assert model.fit_calls == 1


def test_factory_cannot_reuse_previous_fold_model(interactions):
    model = MemorizingModel()
    with pytest.raises(ValueError, match='unfitted model'):
        CrossValidator(2).cross_validate(lambda: model, interactions, metric='Recall@4')
    assert model.fit_calls == 1


@pytest.mark.parametrize('factory', [False, True])
def test_fresh_models_do_not_retain_held_out_knowledge(factory, interactions):
    original = MemorizingModel()
    result = CrossValidator(2).cross_validate(
        MemorizingModel if factory else original, interactions, metric='Recall@4')
    assert result['folds'] == [0., 0.]
    assert result['mean'] == 0.
    assert original.items == set()
    assert original.fit_calls == 0
    assert not original.is_fitted
