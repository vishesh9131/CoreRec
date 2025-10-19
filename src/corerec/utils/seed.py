"""
Seed utilities for CoreRec framework.

This module provides utilities for setting random seeds for reproducibility.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    This function sets the random seed for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make CUDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow seed (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass 