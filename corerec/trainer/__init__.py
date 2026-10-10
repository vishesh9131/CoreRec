"""
Trainer Module

Provides training infrastructure for recommendation models.
"""

from .trainer import Trainer
from .callbacks import Callback, EarlyStopping, ModelCheckpoint

try:
    from .online_trainer import OnlineTrainer
except ImportError:
    OnlineTrainer = None

__all__ = ["Trainer", "Callback", "EarlyStopping", "ModelCheckpoint", "OnlineTrainer"]
