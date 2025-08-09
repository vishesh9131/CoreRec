"""
Core module for CoreRec framework.

This module contains base classes and abstractions for the CoreRec framework.
"""

from corerec.core.base_model import BaseModel
from corerec.core.encoders import AbstractEncoder, TextEncoder, VisionEncoder
from corerec.core.towers import UserTower, ItemTower, TowerFactory
from corerec.core.losses import DotProductLoss, CosineLoss, InfoNCE

__all__ = [
    'BaseModel',
    'AbstractEncoder', 'TextEncoder', 'VisionEncoder',
    'UserTower', 'ItemTower', 'TowerFactory',
    'DotProductLoss', 'CosineLoss', 'InfoNCE'
] 