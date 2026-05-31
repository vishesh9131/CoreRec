"""
CoreRec API versioning and deprecation policy helpers.

See ``docs/source/api_versioning.md`` for the full policy.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

API_VERSION = "0.5"
REMOVAL_VERSION = "1.0"


def deprecated(
    replacement: str,
    *,
    removal_version: str = REMOVAL_VERSION,
    category: type = DeprecationWarning,
) -> Callable[[F], F]:
    """Mark a function or method as deprecated."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__qualname__} is deprecated and will be removed in "
                f"CoreRec {removal_version}. Use {replacement} instead.",
                category,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def warn_deprecated_arg(legacy: str, replacement: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"Argument '{legacy}' is deprecated and will be removed in CoreRec "
        f"{REMOVAL_VERSION}. Use '{replacement}' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
