"""
CoreRec Model Serving Infrastructure

Production-ready model serving with REST API, batch inference, and monitoring.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.serving.model_server import ModelServer, PredictionRequest, RecommendationRequest
from corerec.serving.batch_inference import BatchInferenceEngine
from corerec.serving.model_loader import ModelLoader
from corerec.serving.online import OnlineRecommender

__all__ = [
    "ModelServer",
    "PredictionRequest",
    "RecommendationRequest",
    "BatchInferenceEngine",
    "ModelLoader",
    "OnlineRecommender",
]
