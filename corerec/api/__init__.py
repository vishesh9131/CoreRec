"""
CoreRec Unified API Module

This module provides standardized interfaces for all recommendation models,
CoreRec API Module

Provides the unified base classes, exceptions, and mixins for all recommender models.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

from .base_recommender import BaseRecommender
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

__all__ = [
    "BaseRecommender",
    "TorchRecommender",
    "ModelInterface",  # Deprecated - kept for backward compatibility
    "PredictorInterface",
]
