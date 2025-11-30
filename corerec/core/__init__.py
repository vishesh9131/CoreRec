"""
Core module for CoreRec framework.

This module contains base classes and abstractions for the CoreRec framework.
"""

from corerec.core.base_model import BaseModel
from corerec.core.towers import UserTower, ItemTower, TowerFactory
from corerec.core.losses import DotProductLoss, CosineLoss, InfoNCE

# Optional imports that require transformers
try:
    from corerec.core.encoders import AbstractEncoder, TextEncoder, VisionEncoder

    _ENCODERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _ENCODERS_AVAILABLE = False
    AbstractEncoder = None
    TextEncoder = None
    VisionEncoder = None

__all__ = [
    "BaseModel",
    "UserTower",
    "ItemTower",
    "TowerFactory",
    "DotProductLoss",
    "CosineLoss",
    "InfoNCE",
]

if _ENCODERS_AVAILABLE:
    __all__.extend(["AbstractEncoder", "TextEncoder", "VisionEncoder"])
