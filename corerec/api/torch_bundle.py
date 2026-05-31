"""Helpers for torch production models — safe bundle save/load."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from corerec.api.model_bundle import is_safe_bundle, load_bundle, save_bundle

T = TypeVar("T")


def save_torch_production(
    instance: Any,
    path: Any,
    *,
    config: Dict[str, Any],
    state: Dict[str, Any],
    arrays: Optional[Dict[str, Any]] = None,
    safe: bool = True,
    state_dict: Optional[Dict[str, Any]] = None,
) -> bool:
    """Save using safe bundle. Returns True if saved, False to use legacy path."""
    if not safe:
        return False
    if state_dict is None and getattr(instance, "model", None) is not None:
        state_dict = instance.model.state_dict()
    save_bundle(
        path,
        model_class=f"{instance.__class__.__module__}.{instance.__class__.__name__}",
        config=config,
        state=state,
        state_dict=state_dict,
        arrays=arrays,
    )
    return True


def load_torch_production(
    cls: Type[T],
    path: Any,
    *,
    build_model: Callable[[T, Dict[str, Any]], None],
    factory: Optional[Callable[[Dict[str, Any]], T]] = None,
    restore: Optional[
        Callable[[T, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]], None]
    ] = None,
    map_location: Any = None,
) -> Optional[T]:
    """Load from safe bundle if present. Returns None to fall back to legacy loader."""
    if not is_safe_bundle(path):
        return None
    bundle = load_bundle(path, map_location=map_location)
    cfg = bundle["config"]
    instance = factory(cfg) if factory else cls(**cfg)
    if restore is not None:
        restore(instance, cfg, bundle["state"], bundle.get("arrays"), bundle)
    else:
        for key, val in bundle["state"].items():
            setattr(instance, key, val)
    build_model(instance, bundle)
    if bundle.get("state_dict") is not None and getattr(instance, "model", None) is not None:
        instance.model.load_state_dict(bundle["state_dict"])
        instance.model.eval()
    return instance


def save_numpy_production(
    instance: Any,
    path: Any,
    *,
    config: Dict[str, Any],
    state: Dict[str, Any],
    arrays: Dict[str, Any],
    safe: bool = True,
) -> bool:
    if not safe:
        return False
    save_bundle(
        path,
        model_class=f"{instance.__class__.__module__}.{instance.__class__.__name__}",
        config=config,
        state=state,
        arrays=arrays,
    )
    return True


def load_numpy_production(
    cls: Type[T],
    path: Any,
    *,
    restore: Callable[[T, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], None],
    factory: Optional[Callable[[Dict[str, Any]], T]] = None,
) -> Optional[T]:
    if not is_safe_bundle(path):
        return None
    bundle = load_bundle(path)
    instance = factory(bundle["config"]) if factory else cls(**bundle["config"])
    restore(instance, bundle["config"], bundle["state"], bundle.get("arrays"))
    return instance
