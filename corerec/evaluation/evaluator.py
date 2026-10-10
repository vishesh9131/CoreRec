"""
Model Evaluator

Tools for evaluating and comparing recommendation models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from corerec.evaluation.metrics import RankingMetrics
from corerec.evaluation.evaluate import METRICS, evaluate as run_evaluate

logger = logging.getLogger(__name__)


def _canonical(metric: str) -> str:
    """'ndcg@10', 'NDCG@10', 'hit_rate@5' -> the evaluate() key ('NDCG@10', 'HitRate@5')."""
    name, _, k = metric.partition("@")
    key = name.lower().replace("_", "")
    for canon in METRICS:
        if canon.lower() == key:
            return f"{canon}@{int(k or 10)}"
    raise ValueError(f"unknown metric {metric!r}; use one of {', '.join(METRICS)} with @k")


class Evaluator:
    """
    Model evaluator for recommendation systems.

    Evaluates models on test data using multiple metrics.

    Example::

        from corerec.evaluation import Evaluator

        evaluator = Evaluator(metrics=['NDCG@10', 'MAP@10', 'Recall@20'])

        # Single model evaluation
        results = evaluator.evaluate(model, test_data)

        # Model comparison
        comparison = evaluator.compare_models({
            'NCF': ncf_model,
            'DeepFM': deepfm_model
        }, test_data)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator.

        Args:
            metrics: List of metrics to compute (e.g., ['NDCG@10', 'MAP@10']); any case
                works, and hit_rate / HitRate are the same metric

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.metrics = metrics or ["NDCG@10", "MAP@10", "Precision@10", "Recall@10"]
        for m in self.metrics:
            _canonical(m)  # fail here on a typo, not after scoring every user
        self.ranking_metrics = RankingMetrics()

    def evaluate(self, model, test_data: Dict[Any, List], strict: bool = False,
                 train_interactions=None) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Same protocol and numbers as :func:`corerec.evaluation.evaluate` (what
        ``corerec train`` / ``serve`` / ``retrain`` report), which this calls.

        Args:
            model: Recommendation model with recommend() method
            test_data: Dict mapping user_id to list of relevant items
            strict: re-raise the first per-user error instead of skipping the user
            train_interactions: optional (user, item) pairs or DataFrame; those
                items are removed from each user's list before scoring. Without
                it, excluding seen items is left to the model.

        Returns:
            Each metric under its canonical name (``NDCG@10``, ``Recall@20``,
            ...) and also under the spelling it was requested with, plus
            ``n_users`` (users scored) and ``n_errors`` (users skipped because
            recommend() raised). A metric with no scored users is NaN, not 0.0.
        """
        wanted = {m: _canonical(m) for m in self.metrics}
        pairs = [(u, it) for u, items in test_data.items() for it in items]
        r = run_evaluate(model, pairs, train_interactions=train_interactions,
                         k=sorted({int(c.split("@")[1]) for c in wanted.values()}), strict=strict)
        n_users, n_errors = r["n_users"], r["n_errors"]
        users = len({u for u, _ in pairs})
        if n_errors:
            logger.warning("%d of %d users failed to evaluate", n_errors, users)
        out = {}
        for asked, canon in wanted.items():
            # NaN, not 0.0: a model that crashed on every user must not look like a bad model
            out[canon] = out[asked] = r[canon] if n_users else float("nan")
        out["n_users"], out["n_errors"] = n_users, n_errors
        return out

    def compare_models(
        self, models: Dict[str, Any], test_data: Dict[Any, List]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on same test data.

        Args:
            models: Dict mapping model_name to model instance
            test_data: Test data (user_id -> relevant items)

        Returns:
            Dict mapping model_name to evaluation results

        Example::

            results = evaluator.compare_models({
                'NCF': ncf_model,
                'DeepFM': deepfm_model
            }, test_data)

            # Results:
            # {
            #   'NCF': {'NDCG@10': 0.45, 'MAP@10': 0.38, ...},
            #   'DeepFM': {'NDCG@10': 0.48, 'MAP@10': 0.41, ...}
            # }

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        comparison = {}

        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            results = self.evaluate(model, test_data)
            comparison[model_name] = results

        return comparison

    def generate_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            results: Evaluation results from compare_models()

        Returns:
            Formatted report string

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        report = "=" * 60 + "\n"
        report += "Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"

        # Get all metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        all_metrics = sorted(all_metrics)

        # Create table
        header = f"{'Model':<20}"
        for metric in all_metrics:
            header += f" {metric:>12}"
        report += header + "\n"
        report += "-" * len(header) + "\n"

        for model_name, model_results in results.items():
            row = f"{model_name:<20}"
            for metric in all_metrics:
                value = model_results.get(metric, 0.0)
                row += f" {value:>12.4f}"
            report += row + "\n"

        return report


class CrossValidator:
    """
    Cross-validation utilities.

    Example::

        cv = CrossValidator(n_folds=5)
        out = cv.cross_validate(lambda: SAR(), df, metric='NDCG@10')
        out['mean'], out['folds']

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of folds
            random_state: Random seed

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.n_folds = n_folds
        self.random_state = random_state

    def split(self, data: Any, n_folds: Optional[int] = None) -> List[tuple]:
        """
        Split data into folds.

        Args:
            data: Data to split
            n_folds: Number of folds (uses self.n_folds if None)

        Returns:
            List of (train, test) tuples

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        n_folds = self.n_folds if n_folds is None else n_folds
        import pandas as pd
        from sklearn.model_selection import KFold

        if not isinstance(data, pd.DataFrame):
            raise NotImplementedError("Only DataFrame supported currently")
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        return [(data.iloc[train], data.iloc[test]) for train, test in splitter.split(data)]

    def cross_validate(
        self,
        model,
        data: Any,
        metric: str = "NDCG@10",
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
    ) -> Dict[str, Any]:
        """
        Fit a fresh model on each fold and score it on the held-out rows.

        Args:
            model: zero-arg factory returning an unfitted model, or an unfitted
                model instance (deep-copied per fold so folds don't leak)
            data: DataFrame of interactions
            metric: any metric Evaluator understands, e.g. 'NDCG@10'

        Returns:
            {'mean': float, 'std': float, 'folds': [per-fold score]}

        Raises:
            ValueError: a supplied instance or factory result is already fitted.
        """
        import copy

        if getattr(model, "is_fitted", False):
            raise ValueError("Cross-validation requires an unfitted model; pass a fresh model factory")

        evaluator = Evaluator(metrics=[metric])
        scores = []
        for train, test in self.split(data):
            m = model() if isinstance(model, type) or not hasattr(model, "fit") else copy.deepcopy(model)
            if getattr(m, "is_fitted", False):
                raise ValueError("Cross-validation requires an unfitted model from each factory call")
            m.fit(
                train[user_col].tolist(),
                train[item_col].tolist(),
                train[rating_col].tolist() if rating_col in train else None,
            )
            truth = test.groupby(user_col)[item_col].apply(list).to_dict()
            scores.append(evaluator.evaluate(m, truth)[metric])
        return {"mean": float(np.nanmean(scores)), "std": float(np.nanstd(scores)), "folds": scores}
