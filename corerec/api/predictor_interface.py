"""
Predictor Interface

Common interface for prediction capabilities.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from abc import ABC, abstractmethod
from typing import Any, List


class PredictorInterface(ABC):
    """
    Interface for models with prediction capabilities.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Make a prediction."""
        pass
    
    @abstractmethod
    def predict_proba(self, *args, **kwargs) -> float:
        """Predict probability/score."""
        pass

