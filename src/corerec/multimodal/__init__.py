"""
Multimodal module for CoreRec framework.

This module provides components for multimodal recommendation,
combining text, images, and other modalities.
"""

from corerec.multimodal.fusion_model import (
    MultimodalFusionModel,
    ModalityFusion,
    ConcatFusion,
    AttentionFusion,
    GatingFusion
)

__all__ = [
    'MultimodalFusionModel',
    'ModalityFusion',
    'ConcatFusion',
    'AttentionFusion',
    'GatingFusion'
] 