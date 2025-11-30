"""
Unified Base Recommender Interface

This provides a standardized API that all recommendation models must implement,
ensuring consistency across different algorithm families.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import pandas as pd
from pathlib import Path
import pickle
import json
import copy
import inspect
import warnings
import numpy as np
from datetime import datetime
from glob import glob


class BaseRecommender(ABC):
    """
    Unified base class for ALL recommendation models in CoreRec.

    This is the single source of truth for the recommendation API, replacing
    the deprecated BaseCorerec class. All models should inherit from this class.

    This enforces consistent API across:
    - Collaborative filtering models
    - Content-based models
    - Hybrid models
    - Deep learning models
    - Graph-based models

    Architecture:

    ┌─────────────────────────────────────┐
    │      BaseRecommender                │
    │  (Unified Interface)                │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────▼─────┐      ┌──────▼────┐
    │ PyTorch  │      │ Traditional│
    │ Models   │      │   Models   │
    └────┬─────┘      └──────┬────┘
         │                   │
    ┌────▼────┐      ┌──────▼────┐
    │NCF,DLRM │      │  SVD, ALS │
    │DeepFM...│      │  KNN...   │
    └─────────┘      └───────────┘

    Parameters:
    -----------
    name: str, optional
        Name of the recommender model (default: class name)

    trainable: bool, optional
        Whether the model is trainable (default: True)

    verbose: bool, optional
        Whether to print training logs (default: False)

    Attributes:
    -----------
    num_users: int
        Number of users in training data

    num_items: int
        Number of items in training data

    total_users: int
        Total number of users (including validation/test)

    total_items: int
        Total number of items (including validation/test)

    uid_map: dict
        Mapping of user IDs to indices

    iid_map: dict
        Mapping of item IDs to indices

    max_rating: float
        Maximum value among rating observations

    min_rating: float
        Minimum value among rating observations

    global_mean: float
        Average value over rating observations

    Standard Methods:
        - fit(): Train the model
        - predict(): Score user-item pairs
        - recommend(): Generate top-K recommendations
        - save(): Persist model to disk
        - load(): Load model from disk
        - batch_predict(): Efficient batch scoring
        - batch_recommend(): Efficient batch recommendations

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, name: Optional[str] = None, trainable: bool = True, verbose: bool = False):
        """
        Initialize base recommender.

        Args:
            name: Model name for identification (default: class name)
            trainable: Whether model is trainable
            verbose: Whether to print training logs
        """
        self.name = name if name is not None else self.__class__.__name__
        self.trainable = trainable
        self.verbose = verbose
        self.is_fitted = False
        self._version = "1.0.0"

        # Attributes to be ignored when saving model
        self.ignored_attrs = ["train_set", "val_set", "test_set"]

        # Useful information from train_set for prediction
        self.num_users: Optional[int] = None
        self.num_items: Optional[int] = None
        self.uid_map: Optional[Dict[Any, int]] = None
        self.iid_map: Optional[Dict[Any, int]] = None
        self.max_rating: Optional[float] = None
        self.min_rating: Optional[float] = None
        self.global_mean: Optional[float] = None

        # Private cached user/item lists
        self.__user_ids: Optional[List[Any]] = None
        self.__item_ids: Optional[List[Any]] = None

    @property
    def total_users(self) -> int:
        """Total number of users including users in test and validation if exists."""
        return len(self.uid_map) if self.uid_map is not None else (self.num_users or 0)

    @property
    def total_items(self) -> int:
        """Total number of items including items in test and validation if exists."""
        return len(self.iid_map) if self.iid_map is not None else (self.num_items or 0)

    @property
    def user_ids(self) -> List[Any]:
        """Return the list of raw user IDs."""
        if self.__user_ids is None and self.uid_map is not None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids or []

    @property
    def item_ids(self) -> List[Any]:
        """Return the list of raw item IDs."""
        if self.__item_ids is None and self.iid_map is not None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids or []

    def reset_info(self) -> None:
        """Reset early stopping and training info."""
        self.best_value = -np.inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        """Deep copy implementation that skips ignored attributes."""
        cls = self.__class__
        result = cls.__new__(cls)
        ignored_attrs = set(self.ignored_attrs)
        for k, v in self.__dict__.items():
            if k in ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @classmethod
    def _get_init_params(cls) -> List[str]:
        """Get initial parameters from the model constructor."""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]
        return sorted([p.name for p in parameters])

    def clone(self, new_params: Optional[Dict[str, Any]] = None) -> "BaseRecommender":
        """
        Clone an instance of the model object.

        Args:
            new_params: Optional dict of parameters to override

        Returns:
            New instance of the model with same parameters
        """
        new_params = {} if new_params is None else new_params
        init_params = {}
        for name in self._get_init_params():
            init_params[name] = new_params.get(name, copy.deepcopy(getattr(self, name, None)))
        return self.__class__(**init_params)

    # Backward compatibility methods from BaseCorerec

    def knows_user(self, user_idx: int) -> bool:
        """
        Return whether the model knows user by its index.

        Args:
            user_idx: User index

        Returns:
            True if user is known, False otherwise
        """
        return (
            user_idx is not None
            and user_idx >= 0
            and self.num_users is not None
            and user_idx < self.num_users
        )

    def knows_item(self, item_idx: int) -> bool:
        """
        Return whether the model knows item by its index.

        Args:
            item_idx: Item index

        Returns:
            True if item is known, False otherwise
        """
        return (
            item_idx is not None
            and item_idx >= 0
            and self.num_items is not None
            and item_idx < self.num_items
        )

    def is_unknown_user(self, user_idx: int) -> bool:
        """Return whether the model doesn't know user by its index."""
        return not self.knows_user(user_idx)

    def is_unknown_item(self, item_idx: int) -> bool:
        """Return whether the model doesn't know item by its index."""
        return not self.knows_item(item_idx)

    def monitor_value(self, train_set: Any, val_set: Any) -> Optional[float]:
        """
        Calculate monitored value used for early stopping on validation set.

        Override this method in subclasses to implement custom monitoring.

        Args:
            train_set: Training dataset
            val_set: Validation dataset

        Returns:
            Monitored value (higher is better) or None
        """
        raise NotImplementedError("Subclass must implement monitor_value() for early stopping")

    def early_stop(
        self, train_set: Any, val_set: Any, min_delta: float = 0.0, patience: int = 0
    ) -> bool:
        """
        Check if training should stop when validation loss has stopped improving.

        Args:
            train_set: Training dataset
            val_set: Validation dataset
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs to wait without improvement

        Returns:
            True if training should stop, False otherwise
        """
        self.current_epoch += 1
        current_value = self.monitor_value(train_set, val_set)
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            if self.verbose:
                print("Early stopping:")
                print(f"- best epoch = {self.best_epoch}, stopped epoch = {self.stopped_epoch}")
                print(
                    f"- best monitored value = {self.best_value:.6f} "
                    f"(delta = {current_value - self.best_value:.6f})"
                )
            return True
        return False

    # Abstract methods that must be implemented by subclasses

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseRecommender":
        """
        Train the recommendation model.

        This method must be implemented by all subclasses.

        Returns:
            self: For method chaining

        Example:
            model.fit(train_data, epochs=10).save('model.pkl')

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    @abstractmethod
    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """
        Predict score for a single user-item pair.

        This method must be implemented by all subclasses.

        Args:
            user_id: User identifier
            item_id: Item identifier
            **kwargs: Additional prediction parameters

        Returns:
            Predicted score (higher = more relevant)

        Example:
            score = model.predict(user_id=123, item_id=456)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    @abstractmethod
    def recommend(
        self, user_id: Any, top_k: int = 10, exclude_items: Optional[List[Any]] = None, **kwargs
    ) -> List[Any]:
        """
        Generate top-K item recommendations for a user.

        This method must be implemented by all subclasses.

        Args:
            user_id: User identifier
            top_k: Number of recommendations to generate
            exclude_items: Items to exclude from recommendations
            **kwargs: Additional recommendation parameters

        Returns:
            List of recommended item IDs (sorted by relevance)

        Example:
            recs = model.recommend(user_id=123, top_k=10)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save model to disk.

        This method must be implemented by all subclasses.

        Args:
            path: File path to save model
            **kwargs: Additional save parameters

        Example:
            model.save('models/ncf_model.pkl')

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseRecommender":
        """
        Load model from disk.

        This method must be implemented by all subclasses.

        Args:
            path: File path to load model from

        Returns:
            Loaded model instance

        Example:
            model = NCF.load('models/ncf_model.pkl')

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    # Non-abstract convenience methods

    def batch_predict(self, pairs: List[Tuple[Any, Any]], **kwargs) -> List[float]:
        """
        Predict scores for multiple user-item pairs efficiently.

        Args:
            pairs: List of (user_id, item_id) tuples
            **kwargs: Additional parameters

        Returns:
            List of predicted scores

        Example:
            scores = model.batch_predict([(1,10), (1,11), (2,10)])

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return [self.predict(user_id, item_id, **kwargs) for user_id, item_id in pairs]

    def batch_recommend(
        self, user_ids: List[Any], top_k: int = 10, **kwargs
    ) -> Dict[Any, List[Any]]:
        """
        Generate recommendations for multiple users efficiently.

        Args:
            user_ids: List of user identifiers
            top_k: Number of recommendations per user
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping user_id to list of recommended items

        Example:
            recs = model.batch_recommend([1, 2, 3], top_k=5)
            # {1: [10,11,12,13,14], 2: [20,21,22,23,24], ...}

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return {uid: self.recommend(uid, top_k, **kwargs) for uid in user_ids}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.

        Returns:
            Dictionary containing model info

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return {
            "name": self.name,
            "version": self._version,
            "is_fitted": self.is_fitted,
            "trainable": self.trainable,
            "model_type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "num_users": self.num_users,
            "num_items": self.num_items,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
