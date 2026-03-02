"""
CoreRec utility functions.

Provides validation, logging, training utilities, and diagnostics.
"""

from .validation import (
    ValidationError,
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)

from .logging import setup_logging, get_logger, LoggerAdapter

from .training_utils import EarlyStopping

from .seed import set_seed

from .config import load_config, merge_configs

# Similarity utilities (for SAR and other CF models)
from .similarity import (
    jaccard,
    cosine_similarity,
    lift,
    inclusion_index,
    mutual_information,
    lexicographers_mutual_information,
    exponential_decay,
    get_top_k_scored_items,
    rescale,
)

# Diagnostics (optional - may not be available if NumPy not installed)
try:
    from .diagnostics import (
        check_numpy_backend,
        check_mkl_warning,
        print_system_info,
        fix_mkl_warning_instructions,
    )
except ImportError:
    # NumPy not available, skip diagnostics
    pass

__all__ = [
    # Validation
    "ValidationError",
    "validate_fit_inputs",
    "validate_user_id",
    "validate_top_k",
    "validate_model_fitted",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggerAdapter",
    # Training
    "EarlyStopping",
    # Seed
    "set_seed",
    # Config
    "load_config",
    "merge_configs",
    # Diagnostics
    "check_numpy_backend",
    "check_mkl_warning",
    "print_system_info",
    "fix_mkl_warning_instructions",
    # Similarity functions
    "jaccard",
    "cosine_similarity",
    "lift",
    "inclusion_index",
    "mutual_information",
    "lexicographers_mutual_information",
    "exponential_decay",
    "get_top_k_scored_items",
    "rescale",
]

