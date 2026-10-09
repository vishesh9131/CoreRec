"""
Model Evaluator

Tools for evaluating and comparing recommendation models.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from corerec.evaluation.metrics import RankingMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluator for recommendation systems.

    Evaluates models on test data using multiple metrics.

    Example::

        from corerec.evaluation import Evaluator

        evaluator = Evaluator(metrics=['ndcg@10', 'map@10', 'recall@20'])

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
            metrics: List of metrics to compute (e.g., ['ndcg@10', 'map@10'])

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.metrics = metrics or ["ndcg@10", "map@10", "precision@10", "recall@10"]
        self.ranking_metrics = RankingMetrics()

    def evaluate(self, model, test_data: Dict[Any, List], strict: bool = False) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            model: Recommendation model with recommend() method
            test_data: Dict mapping user_id to list of relevant items
            strict: re-raise the first per-user error instead of skipping the user

        Returns:
            Dictionary of metric_name -> score, plus ``n_users`` (users scored)
            and ``n_errors`` (users skipped because recommend() raised).
            A metric with no scored users is NaN, not 0.0.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        results = {metric: [] for metric in self.metrics}
        n_errors = 0
        # ask for as many items as the deepest metric needs; a fixed 20 made
        # recall@50 count ranks 21..50 as misses
        max_k = max(int(m.split("@")[1]) if "@" in m else 10 for m in self.metrics)

        for user_id, ground_truth in test_data.items():
            try:
                # Get recommendations
                predictions = model.recommend(user_id, top_k=max_k)

                # Compute each metric
                for metric_name in self.metrics:
                    # Parse metric name (e.g., 'ndcg@10')
                    if "@" in metric_name:
                        metric, k = metric_name.split("@")
                        k = int(k)
                    else:
                        metric = metric_name
                        k = 10

                    # Compute metric
                    if metric == "ndcg":
                        score = self.ranking_metrics.ndcg_at_k(predictions, ground_truth, k)
                    elif metric == "map":
                        score = self.ranking_metrics.map_at_k(predictions, ground_truth, k)
                    elif metric == "mrr":
                        score = self.ranking_metrics.mrr_at_k(predictions, ground_truth, k)
                    elif metric == "precision":
                        score = self.ranking_metrics.precision_at_k(predictions, ground_truth, k)
                    elif metric == "recall":
                        score = self.ranking_metrics.recall_at_k(predictions, ground_truth, k)
                    elif metric == "hit_rate":
                        score = self.ranking_metrics.hit_rate_at_k(predictions, ground_truth, k)
                    else:
                        continue

                    results[metric_name].append(score)

            except Exception as e:
                if strict:
                    raise
                n_errors += 1
                logger.warning("Error evaluating user %s: %s", user_id, e)

        if n_errors:
            logger.warning("%d of %d users failed to evaluate", n_errors, len(test_data))
        # NaN, not 0.0: a model that crashed on every user must not look like a bad model
        out = {k: float(np.mean(v)) if v else float("nan") for k, v in results.items()}
        out["n_users"] = len(test_data) - n_errors
        out["n_errors"] = n_errors
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
            #   'NCF': {'ndcg@10': 0.45, 'map@10': 0.38, ...},
            #   'DeepFM': {'ndcg@10': 0.48, 'map@10': 0.41, ...}
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
        out = cv.cross_validate(lambda: SAR(), df, metric='ndcg@10')
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
        n_folds = n_folds or self.n_folds

        # Simple implementation - can be enhanced
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            data = data.sample(frac=1, random_state=self.random_state)  # Shuffle
            fold_size = len(data) // n_folds

            folds = []
            for i in range(n_folds):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < n_folds - 1 else len(data)

                test_data = data.iloc[test_start:test_end]
                train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])

                folds.append((train_data, test_data))

            return folds
        else:
            raise NotImplementedError("Only DataFrame supported currently")

    def cross_validate(
        self,
        model,
        data: Any,
        metric: str = "ndcg@10",
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
            metric: any metric Evaluator understands, e.g. 'ndcg@10'

        Returns:
            {'mean': float, 'std': float, 'folds': [per-fold score]}
        """
        import copy

        evaluator = Evaluator(metrics=[metric])
        scores = []
        for train, test in self.split(data):
            m = model() if isinstance(model, type) or not hasattr(model, "fit") else copy.deepcopy(model)
            m.fit(
                train[user_col].tolist(),
                train[item_col].tolist(),
                train[rating_col].tolist() if rating_col in train else [1.0] * len(train),
            )
            truth = test.groupby(user_col)[item_col].apply(list).to_dict()
            scores.append(evaluator.evaluate(m, truth)[metric])
        return {"mean": float(np.nanmean(scores)), "std": float(np.nanstd(scores)), "folds": scores}
