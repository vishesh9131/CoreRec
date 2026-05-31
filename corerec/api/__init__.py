"""
CoreRec Unified API Module

Provides the unified base classes, exceptions, dataset helpers, and mixins
for all recommender models.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

from .base_recommender import BaseRecommender
from .dataset import RecommenderDataset, coerce_dataset, is_recommender_dataset
from .exceptions import (
    CoreRecException,
    ModelNotFittedError,
    InvalidDataError,
    InvalidParameterError,
    SaveLoadError,
    RecommendationError,
    ConfigurationError,
)
from .mixins import (
    ModelPersistenceMixin,
    BatchProcessingMixin,
    ValidationMixin,
    EarlyStoppingMixin,
)
from .recommend_args import normalize_recommend_kwargs
from .safe_persistence import save_artifact, load_artifact, COREREC_SAVE_VERSION
from .versioning import API_VERSION, REMOVAL_VERSION, deprecated, warn_deprecated_arg

__all__ = [
    "BaseRecommender",
    "RecommenderDataset",
    "coerce_dataset",
    "is_recommender_dataset",
    "CoreRecException",
    "ModelNotFittedError",
    "InvalidDataError",
    "InvalidParameterError",
    "SaveLoadError",
    "RecommendationError",
    "ConfigurationError",
    "ModelPersistenceMixin",
    "BatchProcessingMixin",
    "ValidationMixin",
    "EarlyStoppingMixin",
    "normalize_recommend_kwargs",
    "save_artifact",
    "load_artifact",
    "COREREC_SAVE_VERSION",
    "API_VERSION",
    "REMOVAL_VERSION",
    "deprecated",
    "warn_deprecated_arg",
]
