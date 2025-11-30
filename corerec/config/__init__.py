"""
CoreRec Configuration Management

Unified configuration system for all CoreRec models and components.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.config.config_manager import ConfigManager, ModelConfig
from corerec.config.config_validator import ConfigValidator

__all__ = [
    "ConfigManager",
    "ModelConfig",
    "ConfigValidator",
]
