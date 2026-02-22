"""
Input validation utilities for CoreRec engines.

Provides comprehensive validation for fit(), predict(), and recommend() methods
to give users clear, actionable error messages instead of cryptic crashes.

Author: CoreRec Team
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Custom exception for validation errors with helpful messages."""

    pass


def validate_fit_inputs(
    user_ids: List[Any],
    item_ids: List[Any],
    ratings: Optional[List[float]] = None,
    allow_empty: bool = False,
) -> None:
    """
    Validate inputs for model.fit() method.

    Args:
        user_ids: List of user identifiers
        item_ids: List of item identifiers
        ratings: Optional list of ratings/interactions
        allow_empty: Whether to allow empty lists

    Raises:
        ValidationError: If inputs are invalid with detailed explanation

    Example:
        >>> validate_fit_inputs([1,2,3], [4,5,6], [5.0, 4.0, 3.0])
        >>> # No error - inputs are valid
    """
    # Check for None
    if user_ids is None:
        raise ValidationError(
            "user_ids cannot be None. "
            "Expected a list of user identifiers. "
            "Example: user_ids=[1, 2, 3, 4, 5]"
        )

    if item_ids is None:
        raise ValidationError(
            "item_ids cannot be None. "
            "Expected a list of item identifiers. "
            "Example: item_ids=[10, 20, 30, 40, 50]"
        )

    # Convert to lists if needed
    if not isinstance(user_ids, (list, tuple, np.ndarray, pd.Series)):
        raise ValidationError(
            f"user_ids must be a list, tuple, numpy array, or pandas Series. "
            f"Got {type(user_ids).__name__}. "
            f"Example: user_ids=[1, 2, 3]"
        )

    if not isinstance(item_ids, (list, tuple, np.ndarray, pd.Series)):
        raise ValidationError(
            f"item_ids must be a list, tuple, numpy array, or pandas Series. "
            f"Got {type(item_ids).__name__}. "
            f"Example: item_ids=[10, 20, 30]"
        )

    # Convert to lists for validation
    user_ids = list(user_ids)
    item_ids = list(item_ids)
    if ratings is not None:
        ratings = list(ratings)

    # Check for empty inputs
    if not allow_empty:
        if len(user_ids) == 0:
            raise ValidationError(
                "user_ids cannot be empty. "
                "Please provide at least one user-item interaction. "
                "Example: model.fit(user_ids=[1, 2], item_ids=[10, 20], ratings=[5.0, 4.0])")

        if len(item_ids) == 0:
            raise ValidationError(
                "item_ids cannot be empty. "
                "Please provide at least one user-item interaction. "
                "Example: model.fit(user_ids=[1, 2], item_ids=[10, 20], ratings=[5.0, 4.0])")

    # Check length matching
    if len(user_ids) != len(item_ids):
        raise ValidationError(
            f"Length mismatch: user_ids has {len(user_ids)} elements, "
            f"but item_ids has {len(item_ids)} elements. "
            f"All input lists must have the same length. "
            f"Each user_id[i], item_id[i], rating[i] represents one interaction."
        )

    if ratings is not None:
        if len(ratings) != len(user_ids):
            raise ValidationError(
                f"Length mismatch: user_ids has {len(user_ids)} elements, "
                f"but ratings has {len(ratings)} elements. "
                f"All input lists must have the same length."
            )

        # Check rating types
        try:
            ratings_float = [float(r) for r in ratings]
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"ratings must contain numeric values that can be converted to float. "
                f"Found invalid value. Error: {str(e)}"
            )


def validate_user_id(user_id: Any,
                     user_map: Dict[Any,
                                    int],
                     allow_unknown: bool = False) -> None:
    """
    Validate a single user ID.

    Args:
        user_id: User identifier to validate
        user_map: Dictionary mapping user IDs to indices
        allow_unknown: Whether to allow unknown users

    Raises:
        ValidationError: If user_id is invalid
    """
    if user_id is None:
        raise ValidationError(
            "user_id cannot be None. "
            "Please provide a valid user identifier.")

    if not allow_unknown and user_id not in user_map:
        # Get some example valid user IDs
        example_users = list(user_map.keys())[:5]
        raise ValidationError(
            f"Unknown user_id: {user_id}. "
            f"This user was not seen during training. "
            f"Valid user IDs include: {example_users}... "
            f"Total {len(user_map)} users in training data."
        )


def validate_item_id(item_id: Any,
                     item_map: Dict[Any,
                                    int],
                     allow_unknown: bool = False) -> None:
    """
    Validate a single item ID.

    Args:
        item_id: Item identifier to validate
        item_map: Dictionary mapping item IDs to indices
        allow_unknown: Whether to allow unknown items

    Raises:
        ValidationError: If item_id is invalid
    """
    if item_id is None:
        raise ValidationError(
            "item_id cannot be None. "
            "Please provide a valid item identifier.")

    if not allow_unknown and item_id not in item_map:
        example_items = list(item_map.keys())[:5]
        raise ValidationError(
            f"Unknown item_id: {item_id}. "
            f"This item was not seen during training. "
            f"Valid item IDs include: {example_items}... "
            f"Total {len(item_map)} items in training data."
        )


def validate_top_k(top_k: int, min_val: int = 1, max_val: int = 10000) -> None:
    """
    Validate top_k parameter for recommendations.

    Args:
        top_k: Number of recommendations to return
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        ValidationError: If top_k is invalid
    """
    if not isinstance(top_k, int):
        raise ValidationError(
            # f"top_k must be an integer. Got {type(top_k).__name__}. " f"Example: top_k=10")
            f"top_k must be an integer. Got {type(top_k).__name__}")


    if top_k < min_val:
        raise ValidationError(
            f"top_k must be at least {min_val}. Got {top_k}. "
            f"Example: model.recommend(user_id=1, top_k=10)"
        )

    if top_k > max_val:
        raise ValidationError(
            f"top_k cannot exceed {max_val}. Got {top_k}. "
            f"Consider using a smaller value for better performance."
        )


def validate_model_fitted(is_fitted: bool, model_name: str = "Model") -> None:
    """
    Validate that model has been fitted before prediction/recommendation.

    Args:
        is_fitted: Whether the model has been fitted
        model_name: Name of the model for error message

    Raises:
        ValidationError: If model is not fitted
    """
    if not is_fitted:
        raise ValidationError(
            f"{model_name} has not been trained yet. "
            f"Please call model.fit(user_ids, item_ids, ratings) before making predictions. "
            f"\n\nExample:\n"
            f"    model = {model_name}()\n"
            f"    model.fit(\n"
            f"        user_ids=[1, 2, 3, 4, 5],\n"
            f"        item_ids=[10, 20, 30, 40, 50],\n"
            f"        ratings=[5.0, 4.0, 3.0, 5.0, 2.0]\n"
            f"    )\n"
            f"    recommendations = model.recommend(user_id=1, top_k=10)"
        )


def validate_rating_range(
        ratings: List[float],
        min_rating: float = 0.0,
        max_rating: float = 5.0,
        warn_only: bool = False) -> None:
    """
    Validate that ratings are within expected range.

    Args:
        ratings: List of ratings
        min_rating: Minimum expected rating
        max_rating: Maximum expected rating
        warn_only: If True, only warn instead of raising error

    Raises:
        ValidationError: If ratings are outside range (unless warn_only=True)
    """
    import warnings

    ratings_array = np.array(ratings)
    actual_min = ratings_array.min()
    actual_max = ratings_array.max()

    if actual_min < min_rating or actual_max > max_rating:
        message = (
            f"Ratings outside expected range [{min_rating}, {max_rating}]. "
            f"Found ratings in range [{actual_min:.2f}, {actual_max:.2f}]. "
            f"This may indicate incorrect data or unusual rating scale. "
            f"If your rating scale is different, this warning can be ignored."
        )

        if warn_only:
            warnings.warn(message, UserWarning)
        else:
            raise ValidationError(message)


def validate_embeddings_dim(
        embedding_dim: int,
        min_val: int = 1,
        max_val: int = 1024) -> None:
    """
    Validate embedding dimension parameter.

    Args:
        embedding_dim: Dimension of embeddings
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        ValidationError: If embedding_dim is invalid
    """
    if not isinstance(embedding_dim, int):
        raise ValidationError(f"embedding_dim must be an integer. Got {type(embedding_dim).__name__}")

    if embedding_dim < min_val:
        raise ValidationError(
            f"embedding_dim must be at least {min_val}. Got {embedding_dim}. "
            f"Typical values: 16, 32, 64, 128, 256"
        )

    if embedding_dim > max_val:
        raise ValidationError(
            f"embedding_dim cannot exceed {max_val}. Got {embedding_dim}. "
            f"Large embeddings may cause memory issues. Consider using a smaller value."
        )


