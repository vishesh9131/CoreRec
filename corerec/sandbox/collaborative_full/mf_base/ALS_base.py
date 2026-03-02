from typing import Union
from pathlib import Path
from typing import Optional
from corerec.api.exceptions import ModelNotFittedError
import pickle
import numpy as np

# Alternating Least Squares
class ALSBase:
    def __init__(self, num_factors: int = 100, regularization: float = 0.01, alpha: float = 1.0, iterations: int = 15, seed: Optional[int] = None):
        self.num_factors = num_factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def save(self, path: Union[str, Path], **kwargs) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)