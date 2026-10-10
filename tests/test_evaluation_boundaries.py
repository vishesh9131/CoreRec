import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from corerec.evaluation import CrossValidator, RankingMetrics, evaluate


METRICS = [RankingMetrics.ndcg_at_k, RankingMetrics.map_at_k,
           RankingMetrics.mrr_at_k, RankingMetrics.precision_at_k,
           RankingMetrics.recall_at_k, RankingMetrics.hit_rate_at_k]


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('k', [0, -1, True, 1.5, '2', None])
def test_ranking_metrics_reject_invalid_cutoffs(metric, k):
    with pytest.raises(ValueError, match='positive integer'):
        metric([1, 2], [1], k=k)


@pytest.mark.parametrize('k', [0, -1, True, 1.5, '2', None, [], [1, 0]])
def test_evaluate_rejects_invalid_cutoffs_before_scoring(k):
    model = Mock()
    with pytest.raises(ValueError, match='k must'):
        evaluate(model, [(1, 1)], k=k)
    model.recommend.assert_not_called()


def test_numpy_integer_cutoffs_and_multiple_cutoffs_work():
    model = Mock()
    model.recommend.return_value = [1, 2]
    result = evaluate(model, [(1, 1)], k=[np.int64(2), 1, 2])
    assert result['Precision@1'] == 1.
    assert result['Precision@2'] == .5
    model.recommend.assert_called_once()
    for metric in METRICS:
        assert metric([1, 2], [1], k=np.int64(2)) == metric([1, 2], [1], k=2)
    assert evaluate(model, [(1, 1)], k=np.int64(2))['Precision@2'] == .5


def test_cv_folds_are_balanced_complete_disjoint_and_reproducible():
    frame = pd.DataFrame({'row': range(10)}, index=['same'] * 10)
    cv = CrossValidator(n_folds=6, random_state=7)
    folds = cv.split(frame)
    sizes = [len(test) for train, test in folds]
    assert max(sizes) - min(sizes) <= 1
    assert sorted(row for train, test in folds for row in test['row']) == list(range(10))
    for (train, test), (again_train, again_test) in zip(folds, cv.split(frame)):
        assert set(train['row']).isdisjoint(test['row'])
        assert len(train) + len(test) == len(frame)
        pd.testing.assert_frame_equal(train, again_train)
        pd.testing.assert_frame_equal(test, again_test)


@pytest.mark.parametrize('folds', [0, 1, -1, 11, 2.5])
def test_cv_rejects_invalid_fold_counts_and_explicit_overrides(folds):
    frame = pd.DataFrame({'row': range(10)})
    with pytest.raises(ValueError):
        CrossValidator(n_folds=folds).split(frame)
    with pytest.raises(ValueError):
        CrossValidator(n_folds=3).split(frame, n_folds=folds)


@pytest.mark.parametrize('name', ['SAR', 'DCN', 'DeepFM'])
def test_cross_validate_keeps_explicit_ratings_required(name):
    import corerec.engines as engines
    from corerec.api.exceptions import InvalidDataError

    frame = pd.DataFrame({'user_id': [0, 0, 1, 1], 'item_id': [10, 11, 11, 12]})
    with pytest.raises(InvalidDataError, match='Explicit ratings are required'):
        CrossValidator(n_folds=2).cross_validate(getattr(engines, name), frame)


def test_cross_validate_still_supports_implicit_models_without_ratings():
    from corerec.engines import ItemKNN

    frame = pd.DataFrame({'user_id': [0, 0, 1, 1, 2, 2], 'item_id': [10, 11, 11, 12, 12, 10]})
    result = CrossValidator(n_folds=2).cross_validate(ItemKNN, frame)
    assert len(result['folds']) == 2
    assert np.isfinite(result['mean'])
