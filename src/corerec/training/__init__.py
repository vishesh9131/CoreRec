"""
CoreRec Unified Training Framework

Standardized training infrastructure with callbacks, scheduling, and monitoring.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.training.trainer import Trainer
from corerec.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoardLogger,
)

__all__ = [
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "TensorBoardLogger",
]
