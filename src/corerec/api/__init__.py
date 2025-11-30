"""
CoreRec Unified API Module

This module provides standardized interfaces for all recommendation models,
ensuring consistency across collaborative filtering, content-based, and hybrid systems.
"""

from corerec.api.base_recommender import BaseRecommender
from corerec.api.model_interface import ModelInterface
from corerec.api.predictor_interface import PredictorInterface

__all__ = [
    "BaseRecommender",
    "ModelInterface",
    "PredictorInterface",
]
