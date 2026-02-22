"""
Multimodal module for CoreRec framework.

This module provides components for multimodal recommendation,
combining text, images, and other modalities.
"""

# Optional imports (may not exist yet)
try:
    from corerec.multimodal.fusion_model import (
        MultimodalFusionModel,
        ModalityFusion,
    )
except ImportError:
    MultimodalFusionModel = None
    ModalityFusion = None

# New fusion strategies
try:
    from corerec.multimodal.fusion_strategies import (
        MultiModalFusion,
        ConcatFusion,
        WeightedFusion,
        AttentionFusion,
        GatedFusion,
        BilinearFusion,
        FusionStrategy,
    )
except ImportError:
    MultiModalFusion = None
    ConcatFusion = None
    WeightedFusion = None
    AttentionFusion = None
    GatedFusion = None
    BilinearFusion = None
    FusionStrategy = None

__all__ = [
    "MultimodalFusionModel",
    "ModalityFusion",
    "MultiModalFusion",
    "ConcatFusion",
    "WeightedFusion",
    "AttentionFusion",
    "GatedFusion",
    "BilinearFusion",
    "FusionStrategy",
]
