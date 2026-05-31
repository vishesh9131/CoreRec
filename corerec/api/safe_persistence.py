"""
Safe model persistence helpers (production path).

Prefer ``torch.save(state_dict)`` + JSON metadata over raw pickle for
neural models.  Pickle remains available via ``allow_pickle=True`` for
backward compatibility until CoreRec 1.0.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

from corerec.api.exceptions import SaveLoadError

COREREC_SAVE_VERSION = "1.0"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def save_artifact(
    path: Union[str, Path],
    *,
    state_dict: Optional[Dict[str, Any]] = None,
    sklearn_payload: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    allow_pickle: bool = False,
) -> None:
    """
    Save model artifact in the safe production format.

    Layout:
        ``{path}.pt``       — torch state_dict (if provided)
        ``{path}.skops``    — pickle fallback for sklearn-only models (if allow_pickle)
        ``{path}.meta.json`` — version, class, hyperparams, maps
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "corerec_save_version": COREREC_SAVE_VERSION,
        **(metadata or {}),
    }

    if state_dict is not None:
        try:
            import torch

            torch.save(state_dict, path.with_suffix(".pt"))
            meta["weights_file"] = str(path.with_suffix(".pt").name)
        except ImportError as e:
            raise SaveLoadError("torch is required to save state_dict artifacts") from e

    if sklearn_payload is not None:
        if not allow_pickle:
            warnings.warn(
                "Saving sklearn payload with pickle. Pass allow_pickle=True explicitly "
                "or migrate to state_dict format.",
                UserWarning,
                stacklevel=2,
            )
        with open(path.with_suffix(".skops"), "wb") as f:
            pickle.dump(sklearn_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta["sklearn_file"] = str(path.with_suffix(".skops").name)

    with open(_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_artifact(
    path: Union[str, Path],
    *,
    map_location: Any = None,
) -> Dict[str, Any]:
    """
    Load saved artifact components.

    Returns dict with keys ``state_dict``, ``sklearn_payload``, ``metadata``.
    """
    path = Path(path)
    meta_file = _meta_path(path)
    if not meta_file.exists():
        raise SaveLoadError(
            f"No metadata file at {meta_file}. Expected safe save format "
            f"(.meta.json sidecar). For legacy pickle-only models use model.load()."
        )

    with open(meta_file, encoding="utf-8") as f:
        metadata = json.load(f)

    result: Dict[str, Any] = {"metadata": metadata, "state_dict": None, "sklearn_payload": None}

    weights_name = metadata.get("weights_file")
    if weights_name:
        import torch

        weights_path = path.parent / weights_name
        result["state_dict"] = torch.load(weights_path, map_location=map_location, weights_only=True)

    skops_name = metadata.get("sklearn_file")
    if skops_name:
        skops_path = path.parent / skops_name
        with open(skops_path, "rb") as f:
            result["sklearn_payload"] = pickle.load(f)

    return result
