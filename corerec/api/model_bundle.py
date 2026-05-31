"""
Safe model bundle persistence (production default).

Format ``corerec_safe_v1``::

    {base}.meta.json     — config + JSON-safe state (no pickle)
    {base}.weights.pt    — torch state_dict (weights_only load)
    {base}.arrays.npz    — numpy arrays (compressed)

Legacy formats (``.pt`` checkpoint, ``.pkl``, ``.npy`` pickle) remain loadable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from corerec.api.exceptions import SaveLoadError

COREREC_SAVE_VERSION = "1.0"
SAFE_FORMAT = "corerec_safe_v1"


def artifact_base(path: Union[str, Path]) -> Path:
    """Normalize ``/tmp/model.pt`` → ``/tmp/model``."""
    p = Path(path)
    for ext in (".weights.pt", ".arrays.npz", ".meta.json", ".pt", ".pkl", ".npy"):
        if p.name.endswith(ext):
            return p.with_name(p.name[: -len(ext)])
    return p


def is_safe_bundle(path: Union[str, Path]) -> bool:
    meta = artifact_base(path).with_suffix(".meta.json")
    if not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return data.get("format") == SAFE_FORMAT
    except (json.JSONDecodeError, OSError):
        return False


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, set):
        return [_jsonify(v) for v in sorted(obj, key=str)]
    return str(obj)


def save_bundle(
    path: Union[str, Path],
    *,
    model_class: str,
    config: Dict[str, Any],
    state: Dict[str, Any],
    state_dict: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a safe v1 model bundle. Returns the bundle base path.
    """
    base = artifact_base(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "format": SAFE_FORMAT,
        "corerec_save_version": COREREC_SAVE_VERSION,
        "model_class": model_class,
        "config": _jsonify(config),
        "state": _jsonify(state),
    }

    if state_dict is not None:
        try:
            import torch
        except ImportError as e:
            raise SaveLoadError("torch required to save state_dict bundle") from e
        weights_path = base.with_suffix(".weights.pt")
        torch.save(state_dict, weights_path)
        meta["weights_file"] = weights_path.name

    if arrays:
        try:
            import numpy as np
        except ImportError as e:
            raise SaveLoadError("numpy required to save array bundle") from e
        npz_path = base.with_suffix(".arrays.npz")
        np.savez_compressed(npz_path, **arrays)
        meta["arrays_file"] = npz_path.name

    meta_path = base.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return base


def load_bundle(path: Union[str, Path], *, map_location: Any = None) -> Dict[str, Any]:
    """Load a safe v1 bundle. Returns dict with config, state, state_dict, arrays."""
    base = artifact_base(path)
    meta_path = base.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise SaveLoadError(f"Safe bundle metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != SAFE_FORMAT:
        raise SaveLoadError(f"Unsupported bundle format: {meta.get('format')!r}")

    result: Dict[str, Any] = {
        "metadata": meta,
        "config": meta.get("config", {}),
        "state": meta.get("state", {}),
        "state_dict": None,
        "arrays": None,
    }

    weights_name = meta.get("weights_file")
    if weights_name:
        import torch

        weights_path = base.parent / weights_name
        result["state_dict"] = torch.load(
            weights_path, map_location=map_location, weights_only=True
        )

    arrays_name = meta.get("arrays_file")
    if arrays_name:
        import numpy as np

        npz_path = base.parent / arrays_name
        with np.load(npz_path, allow_pickle=False) as npz:
            result["arrays"] = {k: npz[k] for k in npz.files}

    return result


def save_legacy_pickle(path: Union[str, Path], payload: Any) -> None:
    """Explicit opt-in legacy pickle (discouraged in production)."""
    import pickle
    import warnings

    warnings.warn(
        "Saving with pickle is deprecated for production. Use safe=True (default).",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
