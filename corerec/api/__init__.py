"""
CoreRec Unified API Module

This module provides standardized interfaces for all recommendation models,
ensuring consistency across collaborative filtering, content-based, and hybrid systems.

Main Classes:
- BaseRecommender: Abstract base for all recommenders (fit, predict, recommend API)
- TorchRecommender: PyTorch-based recommender combining BaseRecommender + BaseModel
- PredictorInterface: Interface for prediction services
- ModelInterface: DEPRECATED - use BaseModel or TorchRecommender instead

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.api.base_recommender import BaseRecommender
from corerec.api.torch_recommender import TorchRecommender
from corerec.api.model_interface import ModelInterface  # Deprecated
from corerec.api.predictor_interface import PredictorInterface

__all__ = [
    "BaseRecommender",
    "TorchRecommender",
    "ModelInterface",  # Deprecated - kept for backward compatibility
    "PredictorInterface",
]

