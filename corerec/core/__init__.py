"""
Core module for CoreRec framework.

This module contains base classes and abstractions for the CoreRec framework.
"""

from corerec.core.base_model import BaseModel
from corerec.core.towers import UserTower, ItemTower, TowerFactory
from corerec.core.losses import DotProductLoss, CosineLoss, InfoNCE

# transformers is only needed once an encoder is built (corerec[transformers])
from corerec.core.encoders import AbstractEncoder, TextEncoder, VisionEncoder

__all__ = [
    "BaseModel",
    "UserTower",
    "ItemTower",
    "TowerFactory",
    "DotProductLoss",
    "CosineLoss",
    "InfoNCE",
    "AbstractEncoder",
    "TextEncoder",
    "VisionEncoder",
]
