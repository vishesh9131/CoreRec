"""
Reusable Mixins for Recommendation Models

This module provides common functionality that can be mixed into
recommendation models to reduce code duplication.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime

from .exceptions import SaveLoadError, ModelNotFittedError

logger = logging.getLogger(__name__)


class ModelPersistenceMixin:
    """
    Mixin providing standardized save/load functionality.

    Usage:
        class MyModel(BaseRecommender, ModelPersistenceMixin):
            pass
    """

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model to disk using pickle.

        Args:
            path: File path to save model (e.g., 'models/my_model.pkl')
            metadata: Optional metadata dict to save alongside model

        Raises:
            SaveLoadError: If save operation fails
        """
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            with open(path_obj, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "model_class": self.__class__.__name__,
                    "saved_at": datetime.now().isoformat(),
                    "is_fitted": getattr(self, "is_fitted", False),
                }
            )

            meta_path = path_obj.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            if getattr(self, "verbose", False):
                logger.info(f"Model saved to {path}")

        except Exception as e:
            raise SaveLoadError(f"Failed to save model to {path}: {e}") from e

    @classmethod
    def load(cls, path: str) -> "ModelPersistenceMixin":
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            Loaded model instance

        Raises:
            SaveLoadError: If load operation fails
        """
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)

            if getattr(model, "verbose", False):
                logger.info(f"Model loaded from {path}")

            return model

        except Exception as e:
            raise SaveLoadError(f"Failed to load model from {path}: {e}") from e


class BatchProcessingMixin:
    """
    Mixin providing efficient batch prediction/recommendation.

    Note: Requires the model to have predict() and recommend() methods.
    """

    def batch_predict(self, pairs: List[Tuple[Any, Any]], **kwargs) -> List[float]:
        """
        Predict scores for multiple user-item pairs.

        Args:
            pairs: List of (user_id, item_id) tuples
            **kwargs: Additional parameters passed to predict()

        Returns:
            List of predicted scores
        """
        if not getattr(self, "is_fitted", False):
            raise ModelNotFittedError()

        return [self.predict(user_id, item_id, **kwargs) for user_id, item_id in pairs]

    def batch_recommend(
        self, user_ids: List[Any], top_k: int = 10, **kwargs
    ) -> Dict[Any, List[Any]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user identifiers
            top_k: Number of recommendations per user
            **kwargs: Additional parameters passed to recommend()

        Returns:
            Dictionary mapping user_id to list of recommended items
        """
        if not getattr(self, "is_fitted", False):
            raise ModelNotFittedError()

        return {uid: self.recommend(uid, top_k, **kwargs) for uid in user_ids}


class ValidationMixin:
    """
    Mixin providing common data validation helpers.
    """

    def _check_is_fitted(self) -> None:
        """
        Check if model is fitted, raise error if not.

        Raises:
            ModelNotFittedError: If model is not fitted
        """
        if not getattr(self, "is_fitted", False):
            raise ModelNotFittedError(
                f"{self.__class__.__name__} must be fitted before use. Call fit() first."
            )

    def _validate_user_id(self, user_id: Any) -> bool:
        """
        Validate if user_id is known to the model.

        Args:
            user_id: User identifier to validate

        Returns:
            True if user is known, False otherwise
        """
        uid_map = getattr(self, "uid_map", None)
        if uid_map is None:
            return True  # No validation possible
        return user_id in uid_map

    def _validate_item_id(self, item_id: Any) -> bool:
        """
        Validate if item_id is known to the model.

        Args:
            item_id: Item identifier to validate

        Returns:
            True if item is known, False otherwise
        """
        iid_map = getattr(self, "iid_map", None)
        if iid_map is None:
            return True  # No validation possible
        return item_id in iid_map


class EarlyStoppingMixin:
    """
    Mixin providing early stopping functionality during training.
    """

    def init_early_stopping(self) -> None:
        """Initialize early stopping state."""
        self.best_value = float("-inf")
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def check_early_stop(
        self, current_value: float, min_delta: float = 0.0, patience: int = 0
    ) -> bool:
        """
        Check if training should stop based on validation metric.

        Args:
            current_value: Current validation metric value
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs to wait for improvement

        Returns:
            True if training should stop, False otherwise
        """
        self.current_epoch += 1

        if current_value - self.best_value >= min_delta:
            # Improvement
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            # No improvement
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            if getattr(self, "verbose", False):
                logger.info(f"Early stopping at epoch {self.stopped_epoch}")
                logger.info(f"Best epoch: {self.best_epoch}, best value: {self.best_value:.6f}")
            return True

        return False
