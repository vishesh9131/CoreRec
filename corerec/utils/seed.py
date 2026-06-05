"""
Seed utilities for CoreRec framework.

This module provides utilities for setting random seeds for reproducibility.
"""

import random
import numpy as np
import os


def set_seed(seed: int):
    """Set random seed for reproducibility.

    This function sets the random seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA), if installed
    - CUDNN

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch is imported lazily so that importing this module (and therefore
    # corerec.utils) does not drag torch (~500 MB) into lightweight CF-only
    # usage such as SAR.
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # TensorFlow seed (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except BaseException:
        pass
