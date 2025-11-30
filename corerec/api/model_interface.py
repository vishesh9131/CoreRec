"""
Model Interface for PyTorch-based Models

DEPRECATED: This module is deprecated. Use BaseModel instead.

For new code:
- Use `corerec.core.base_model.BaseModel` for PyTorch nn.Module models
- Use `corerec.api.torch_recommender.TorchRecommender` for full recommender API

This module is kept for backward compatibility only.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import warnings
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any


class ModelInterface(nn.Module, ABC):
    """
    DEPRECATED: Use BaseModel instead.

    Interface for PyTorch-based recommendation models.

    This class is deprecated and kept only for backward compatibility.
    New models should inherit from:
    - `corerec.core.base_model.BaseModel` for nn.Module functionality
    - `corerec.api.torch_recommender.TorchRecommender` for full API

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "ModelInterface is deprecated. Use BaseModel or TorchRecommender instead. "
            "See corerec/api/README.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by PyTorch models."""
        pass

    def get_embedding_dim(self) -> int:
        """Get embedding dimension if applicable."""
        return getattr(self, "embedding_dim", None)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
