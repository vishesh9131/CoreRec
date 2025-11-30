"""
Towers module for CoreRec framework.

This module provides various tower architectures for encoding different types of data
in recommendation systems.
"""

from corerec.towers.base_tower import AbstractTower
from corerec.towers.mlp_tower import MLPTower
from corerec.towers.transformer_tower import TransformerTower, BERTTower, RoBERTaTower, T5Tower
from corerec.towers.cnn_tower import CNNTower, AttentionPool2d
from corerec.towers.fusion_tower import (
    FusionTower,
    ConcatFusion,
    AttentionFusion,
    GatingFusion,
    SumFusion,
    BilinearFusion,
)

__all__ = [
    # Base classes
    "AbstractTower",
    # MLP Tower
    "MLPTower",
    # Transformer Towers
    "TransformerTower",
    "BERTTower",
    "RoBERTaTower",
    "T5Tower",
    # CNN Tower
    "CNNTower",
    "AttentionPool2d",
    # Fusion Tower
    "FusionTower",
    "ConcatFusion",
    "AttentionFusion",
    "GatingFusion",
    "SumFusion",
    "BilinearFusion",
]
