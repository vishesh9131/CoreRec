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
from .bundle_helpers import (
    dict_from_pairs,
    dense_to_sparse_csr,
    load_map_state,
    nested_dict_from_lists,
    nested_lists,
    pairs,
    save_map_state,
    sparse_to_dense,
    tensor_to_numpy,
)
from .model_bundle import (
    artifact_base,
    is_safe_bundle,
    save_bundle,
    load_bundle,
    SAFE_FORMAT,
)
from .torch_bundle import (
    save_torch_production,
    load_torch_production,
    save_numpy_production,
    load_numpy_production,
)
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
    "artifact_base",
    "is_safe_bundle",
    "save_bundle",
    "load_bundle",
    "SAFE_FORMAT",
    "save_torch_production",
    "load_torch_production",
    "save_numpy_production",
    "load_numpy_production",
    "API_VERSION",
    "REMOVAL_VERSION",
    "deprecated",
    "warn_deprecated_arg",
]
