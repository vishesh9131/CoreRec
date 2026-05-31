"""
Unified recommend() argument normalization for CoreRec production models.

All production models should accept ``top_k`` and ``exclude_items``.
Legacy ``top_n``, ``exclude_seen``, and ``items_to_ignore`` are supported
with DeprecationWarning (removed in CoreRec 1.0).
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

_DEPRECATION_MSG = (
    "{legacy} is deprecated and will be removed in CoreRec 1.0. "
    "Use {replacement} instead."
)


def normalize_recommend_kwargs(
    top_k: int = 10,
    top_n: Optional[int] = None,
    exclude_items: Optional[List[Any]] = None,
    exclude_seen: Optional[bool] = None,
    items_to_ignore: Optional[List[Any]] = None,
    **kwargs: Any,
) -> Tuple[int, Optional[List[Any]], dict]:
    """
    Normalize recommend() keyword arguments to ``top_k`` + ``exclude_items``.

    Returns:
        (top_k, exclude_items, remaining_kwargs)
    """
    if top_n is not None:
        warnings.warn(
            _DEPRECATION_MSG.format(legacy="top_n", replacement="top_k"),
            DeprecationWarning,
            stacklevel=3,
        )
        top_k = top_n

    if items_to_ignore is not None:
        warnings.warn(
            _DEPRECATION_MSG.format(legacy="items_to_ignore", replacement="exclude_items"),
            DeprecationWarning,
            stacklevel=3,
        )
        exclude_items = _merge_exclude(exclude_items, items_to_ignore)

    if exclude_seen is not None:
        warnings.warn(
            _DEPRECATION_MSG.format(
                legacy="exclude_seen (pass seen items via exclude_items)",
                replacement="exclude_items",
            ),
            DeprecationWarning,
            stacklevel=3,
        )
        # exclude_seen=True is default behaviour for most models; False means no masking
        if exclude_seen is False and exclude_items is None:
            exclude_items = []

    if top_k <= 0:
        from corerec.api.exceptions import InvalidParameterError

        raise InvalidParameterError(f"top_k must be a positive integer, got {top_k}")

    return top_k, exclude_items, kwargs


def _merge_exclude(
    primary: Optional[List[Any]], secondary: Optional[List[Any]]
) -> List[Any]:
    merged: List[Any] = []
    if primary:
        merged.extend(primary)
    if secondary:
        merged.extend(secondary)
    return merged
