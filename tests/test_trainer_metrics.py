"""corerec.trainer.metrics agree with corerec.evaluation and scikit-learn (#79)."""

import numpy as np
import pytest
import torch

from corerec.evaluation.metrics import RankingMetrics as R
from corerec.trainer import metrics as M

rng = np.random.default_rng(0)
SCORES = torch.tensor(rng.normal(size=(6, 12)), dtype=torch.float32)
LABELS = torch.tensor((rng.random((6, 12)) < 0.25).astype(np.float32))
LABELS[0] = 0  # a user with nothing relevant


def _lists(i, k):
    ranked = torch.argsort(SCORES[i], descending=True).tolist()[:k]
    truth = torch.nonzero(LABELS[i]).flatten().tolist()
    return ranked, truth


@pytest.mark.parametrize("ours,theirs", [
    (M.precision_at_k, R.precision_at_k), (M.recall_at_k, R.recall_at_k),
    (M.ndcg_at_k, R.ndcg_at_k), (M.hit_rate_at_k, R.hit_rate_at_k),
    (M.average_precision_at_k, R.map_at_k),
])
@pytest.mark.parametrize("k", [3, 5, 20])  # 20 > n_items
def test_ranking_metrics_match_corerec_evaluation(ours, theirs, k):
    expected = np.mean([theirs(*_lists(i, k), k=k) for i in range(len(SCORES))])
    assert float(ours(SCORES, LABELS, k)) == pytest.approx(expected, abs=1e-6)


def test_mrr_is_one_over_the_first_relevant_rank():
    expected = np.mean([R.mrr_at_k(*_lists(i, 12), k=12) for i in range(len(SCORES))])
    assert float(M.mean_reciprocal_rank(SCORES, LABELS)) == pytest.approx(expected, abs=1e-6)
    one = torch.tensor([[0.9, 0.8, 0.1]])
    assert float(M.mean_reciprocal_rank(one, torch.tensor([[1.0, 1.0, 0.0]]))) == 1.0  # not 0.75


def test_ap_with_k_larger_than_the_catalogue_does_not_crash():
    assert float(M.average_precision_at_k(torch.tensor([[0.3, 0.2]]), torch.tensor([[1.0, 0.0]]), k=10)) == 1.0


def test_auc_matches_sklearn_and_handles_ties():
    sk = pytest.importorskip("sklearn.metrics")
    s = torch.tensor(rng.normal(size=200), dtype=torch.float32)
    y = torch.tensor((rng.random(200) < 0.3).astype(np.float32))
    assert float(M.auc_roc(s, y)) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-6)
    tied = torch.tensor([0.5, 0.5, 0.5, 0.5])
    assert float(M.auc_roc(tied, torch.tensor([1.0, 0.0, 1.0, 0.0]))) == pytest.approx(0.5)
    assert float(M.auc_roc(s, torch.zeros(200))) == 0.5


def test_classification_metrics_and_registry():
    logits = torch.tensor([3.0, -3.0, 2.0, -1.0])
    y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert float(M.binary_accuracy(logits, y)) == 0.75
    assert float(M.f1_score(logits, y)) == pytest.approx(2 / 3)
    assert set(M.get_metrics_dict(5)) == {"precision", "recall", "mrr", "ndcg", "hit_rate",
                                         "map", "accuracy", "auc", "f1"}
