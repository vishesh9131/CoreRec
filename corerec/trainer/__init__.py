"""
Trainer Module

Provides training infrastructure for recommendation models.
"""

try:
    from .trainer import Trainer
except ImportError:
    Trainer = None

try:
    from .callbacks import CallbackHandler
except ImportError:
    CallbackHandler = None

try:
    from .online_trainer import OnlineTrainer
except ImportError:
    OnlineTrainer = None

__all__ = ["Trainer", "CallbackHandler", "OnlineTrainer"]
