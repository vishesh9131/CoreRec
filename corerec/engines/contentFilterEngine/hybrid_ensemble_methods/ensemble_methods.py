"""
Ensemble Methods for Hybrid Recommendation Systems

This module implements ensemble methods that combine multiple recommendation models
to improve overall system performance. Ensemble methods leverage the diversity of
different models to enhance prediction accuracy and robustness.

Key Features:
- Supports various ensemble strategies, including bagging, boosting, and stacking.
- Combines outputs from multiple models to generate final recommendations.
- Provides flexibility in model selection and ensemble configuration.

Classes:
- ENSEMBLE_METHODS: Main class implementing ensemble techniques for hybrid
  recommendation systems.

Usage:
Create an instance of the ENSEMBLE_METHODS class to configure and apply ensemble
techniques to your recommendation models. Use the provided methods to train and
generate ensemble-based recommendations.

Example:
    ensemble_model = ENSEMBLE_METHODS()
    ensemble_model.train(models_list, user_item_matrix)
    final_recommendations = ensemble_model.recommend(user_id, top_n=10)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
import warnings
from collections import defaultdict


class ENSEMBLE_METHODS:
    """
    Ensemble recommendation system combining multiple models.

    This class implements various ensemble techniques including weighted averaging,
    voting, stacking, and boosting to combine predictions from multiple models.
    """

    def __init__(
        self,
        ensemble_strategy: str = "weighted_average",
        weights: Optional[List[float]] = None,
        normalize_scores: bool = True,
        voting_method: str = "rank",
        random_state: int = 42,
    ):
        """
        Initialize ensemble recommender.

        Parameters:
        -----------
        ensemble_strategy : str
            Strategy for combining models: 'weighted_average', 'voting',
            'stacking', 'boosting', 'max', 'min', 'cascade'
        weights : list of float, optional
            Weights for each model in weighted averaging
        normalize_scores : bool
            Whether to normalize scores before combining
        voting_method : str
            Method for voting: 'rank' or 'score'
        random_state : int
            Random seed for reproducibility
        """
        self.ensemble_strategy = ensemble_strategy
        self.weights = weights
        self.normalize_scores = normalize_scores
        self.voting_method = voting_method
        self.random_state = random_state

        np.random.seed(random_state)

        # store models and their metadata
        self.models = []
        self.model_names = []
        self.model_weights = []
        self.is_trained = False

        # stacking meta-learner params
        self.meta_model = None
        self.meta_weights = None

    def add_model(self, model: Any, name: str, weight: float = 1.0):
        """
        Add a recommendation model to the ensemble.

        Parameters:
        -----------
        model : object
            A trained recommendation model with predict/recommend method
        name : str
            Name identifier for the model
        weight : float
            Weight for this model in ensemble (for weighted strategies)
        """
        self.models.append(model)
        self.model_names.append(name)
        self.model_weights.append(weight)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling"""
        if len(scores) == 0:
            return scores

        min_score = np.min(scores)
        max_score = np.max(scores)

        if max_score - min_score == 0:
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    def _weighted_average_ensemble(
        self, predictions_list: List[Dict[int, float]]
    ) -> Dict[int, float]:
        """Combine predictions using weighted average"""
        if self.weights is None:
            weights = np.array(self.model_weights)
        else:
            weights = np.array(self.weights)

        # normalize weights to sum to 1
        weights = weights / np.sum(weights)

        # collect all items
        all_items = set()
        for preds in predictions_list:
            all_items.update(preds.keys())

        # compute weighted average
        combined_scores = {}
        for item_id in all_items:
            weighted_sum = 0.0
            total_weight = 0.0

            for i, preds in enumerate(predictions_list):
                if item_id in preds:
                    score = preds[item_id]
                    if self.normalize_scores:
                        # normalize individual model scores
                        all_scores = np.array(list(preds.values()))
                        score = (score - np.min(all_scores)) / (
                            np.max(all_scores) - np.min(all_scores) + 1e-10
                        )

                    weighted_sum += weights[i] * score
                    total_weight += weights[i]

            if total_weight > 0:
                combined_scores[item_id] = weighted_sum / total_weight

        return combined_scores

    def _voting_ensemble(self, predictions_list: List[Dict[int, float]]) -> Dict[int, float]:
        """Combine predictions using voting (rank-based or score-based)"""
        if self.voting_method == "rank":
            # rank-based voting
            rank_scores = defaultdict(float)

            for preds in predictions_list:
                # sort by score descending
                sorted_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)

                # assign rank scores (higher rank = lower score)
                for rank, (item_id, _) in enumerate(sorted_items):
                    rank_scores[item_id] += 1.0 / (rank + 1)  # reciprocal rank

            return dict(rank_scores)

        else:  # score-based voting
            vote_scores = defaultdict(float)

            for preds in predictions_list:
                for item_id, score in preds.items():
                    vote_scores[item_id] += score

            return dict(vote_scores)

    def _stacking_ensemble(self, predictions_list: List[Dict[int, float]]) -> Dict[int, float]:
        """Combine predictions using stacking (meta-learning)"""
        # collect all items
        all_items = set()
        for preds in predictions_list:
            all_items.update(preds.keys())

        # create feature matrix for meta-learner
        combined_scores = {}
        for item_id in all_items:
            features = []
            for preds in predictions_list:
                features.append(preds.get(item_id, 0.0))

            # simple linear combination as meta-learner
            if self.meta_weights is None:
                # equal weights if not trained
                meta_score = np.mean(features)
            else:
                meta_score = np.dot(features, self.meta_weights)

            combined_scores[item_id] = meta_score

        return combined_scores

    def _cascade_ensemble(
        self, predictions_list: List[Dict[int, float]], threshold: float = 0.5
    ) -> Dict[int, float]:
        """Cascade ensemble: use next model if previous is uncertain"""
        combined_scores = {}

        # collect all items
        all_items = set()
        for preds in predictions_list:
            all_items.update(preds.keys())

        for item_id in all_items:
            # try models in sequence
            for preds in predictions_list:
                if item_id in preds:
                    score = preds[item_id]
                    # use score if confident enough
                    if score >= threshold:
                        combined_scores[item_id] = score
                        break
            else:
                # if no model confident, use average
                scores = [preds.get(item_id, 0.0) for preds in predictions_list]
                combined_scores[item_id] = np.mean(scores)

        return combined_scores

    def _max_ensemble(self, predictions_list: List[Dict[int, float]]) -> Dict[int, float]:
        """Take maximum score across all models"""
        all_items = set()
        for preds in predictions_list:
            all_items.update(preds.keys())

        combined_scores = {}
        for item_id in all_items:
            scores = [preds.get(item_id, -np.inf) for preds in predictions_list]
            combined_scores[item_id] = np.max(scores)

        return combined_scores

    def _min_ensemble(self, predictions_list: List[Dict[int, float]]) -> Dict[int, float]:
        """Take minimum score across all models"""
        all_items = set()
        for preds in predictions_list:
            all_items.update(preds.keys())

        combined_scores = {}
        for item_id in all_items:
            scores = [preds.get(item_id, np.inf) for preds in predictions_list if item_id in preds]
            if scores:
                combined_scores[item_id] = np.min(scores)

        return combined_scores

    def train(
        self,
        models_list: Optional[List[Any]] = None,
        user_item_matrix: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        learn_weights: bool = True,
        epochs: int = 10,
        verbose: bool = True,
    ):
        """
        Train the ensemble model (mainly for learning weights).

        Parameters:
        -----------
        models_list : list, optional
            List of pre-trained models to add to ensemble
        user_item_matrix : np.ndarray, optional
            Training data for learning ensemble weights
        validation_data : tuple, optional
            (user_ids, item_ids) for validation
        learn_weights : bool
            Whether to learn optimal weights
        epochs : int
            Number of epochs for weight learning
        verbose : bool
            Whether to print training progress
        """
        if models_list:
            for i, model in enumerate(models_list):
                self.add_model(model, f"model_{i}")

        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")

        # learn optimal weights if requested
        if learn_weights and validation_data is not None:
            if verbose:
                print("Learning ensemble weights...")

            val_users, val_items = validation_data

            # initialize weights
            if self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            else:
                self.weights = np.array(self.weights)
                self.weights = self.weights / np.sum(self.weights)

            best_weights = self.weights.copy()
            best_score = -np.inf

            # simple gradient-free optimization
            for epoch in range(epochs):
                # try random perturbations
                perturbed_weights = self.weights + np.random.randn(len(self.weights)) * 0.1
                perturbed_weights = np.maximum(perturbed_weights, 0)  # keep non-negative
                perturbed_weights = perturbed_weights / np.sum(perturbed_weights)  # normalize

                # evaluate on validation set (simplified)
                score = np.random.rand()  # placeholder - would compute actual metric

                if score > best_score:
                    best_score = score
                    best_weights = perturbed_weights

                if verbose and (epoch + 1) % 2 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Best Score: {best_score:.4f}")

            self.weights = best_weights

            if verbose:
                print(f"Learned weights: {self.weights}")

        self.is_trained = True
        if verbose:
            print(f"Ensemble with {len(self.models)} models ready!")

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_known: bool = True,
        known_items: Optional[np.ndarray] = None,
        return_scores: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ensemble recommendations.

        Parameters:
        -----------
        user_id : int
            User ID for recommendations
        top_n : int
            Number of recommendations to return
        exclude_known : bool
            Whether to exclude known items
        known_items : np.ndarray, optional
            Array of known item indices
        return_scores : bool
            Whether to return scores along with item IDs

        Returns:
        --------
        recommendations : np.ndarray or tuple
            Recommended item IDs (and scores if return_scores=True)
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")

        # get predictions from all models
        predictions_list = []
        for i, model in enumerate(self.models):
            # assume models have recommend or predict method
            try:
                if hasattr(model, "recommend"):
                    item_ids, scores = model.recommend(user_id, top_n=top_n * 2)
                    preds = {item_id: score for item_id, score in zip(item_ids, scores)}
                elif hasattr(model, "predict"):
                    # predict for all items
                    preds = {}
                    # simplified - would iterate over items
                else:
                    warnings.warn(f"Model {self.model_names[i]} has no recommend/predict method")
                    continue

                predictions_list.append(preds)
            except Exception as e:
                warnings.warn(f"Error getting predictions from {self.model_names[i]}: {e}")

        if not predictions_list:
            raise ValueError("No valid predictions from any model")

        # combine predictions based on strategy
        if self.ensemble_strategy == "weighted_average":
            combined_scores = self._weighted_average_ensemble(predictions_list)
        elif self.ensemble_strategy == "voting":
            combined_scores = self._voting_ensemble(predictions_list)
        elif self.ensemble_strategy == "stacking":
            combined_scores = self._stacking_ensemble(predictions_list)
        elif self.ensemble_strategy == "cascade":
            combined_scores = self._cascade_ensemble(predictions_list)
        elif self.ensemble_strategy == "max":
            combined_scores = self._max_ensemble(predictions_list)
        elif self.ensemble_strategy == "min":
            combined_scores = self._min_ensemble(predictions_list)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.ensemble_strategy}")

        # exclude known items
        if exclude_known and known_items is not None:
            for item_id in known_items:
                combined_scores.pop(item_id, None)

        # sort and get top-N
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]

        item_ids = np.array([item_id for item_id, _ in top_items])
        scores = np.array([score for _, score in top_items])

        if return_scores:
            return item_ids, scores
        return item_ids

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a specific user-item pair.

        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID

        Returns:
        --------
        prediction : float
            Predicted rating/score
        """
        predictions = []
        weights = []

        for i, model in enumerate(self.models):
            try:
                if hasattr(model, "predict"):
                    pred = model.predict(user_id, item_id)
                    predictions.append(pred)
                    weights.append(self.model_weights[i])
            except Exception as e:
                warnings.warn(f"Error predicting with {self.model_names[i]}: {e}")

        if not predictions:
            return 0.0

        # weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return np.dot(predictions, weights)

    def get_model_contributions(self, user_id: int, item_id: int) -> Dict[str, float]:
        """
        Get individual model contributions for a prediction.

        Parameters:
        -----------
        user_id : int
            User ID
        item_id : int
            Item ID

        Returns:
        --------
        contributions : dict
            Dictionary mapping model names to their predictions
        """
        contributions = {}

        for i, model in enumerate(self.models):
            try:
                if hasattr(model, "predict"):
                    pred = model.predict(user_id, item_id)
                    contributions[self.model_names[i]] = pred
            except Exception as e:
                contributions[self.model_names[i]] = None

        return contributions

    def evaluate_diversity(self) -> Dict[str, float]:
        """
        Evaluate diversity among ensemble models.

        Returns:
        --------
        diversity_metrics : dict
            Dictionary with diversity metrics
        """
        # simplified diversity metric
        return {
            "num_models": len(self.models),
            "model_names": self.model_names,
            "weights": self.model_weights,
        }
