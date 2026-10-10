"""
Safe model persistence helpers (production path).

Neural weights use numeric bundles without pickle.  Pickle remains available via ``allow_pickle=True`` for
backward compatibility until CoreRec 1.0.
"""

from __future__ import annotations

import json
import pickle
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
        ``{path}.<generation>.weights.npz`` — numeric tensor bytes (if provided)
        ``{path}.skops``    — pickle fallback for sklearn-only models (if allow_pickle)
        ``{path}.meta.json`` — version, class, hyperparams, maps
    """
    if sklearn_payload is not None and not allow_pickle:
        raise SaveLoadError("Saving a Python payload requires allow_pickle=True explicitly")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "corerec_save_version": COREREC_SAVE_VERSION,
        **(metadata or {}),
    }

    if state_dict is not None:
        from corerec.api.model_bundle import save_bundle
        save_bundle(path, model_class="corerec.api.safe_persistence.Artifact",
                    config=meta, state={}, state_dict=state_dict)
        if sklearn_payload is None:
            return
        from corerec.api.model_bundle import bundle_meta_path
        bundle = json.loads(bundle_meta_path(path).read_text(encoding="utf-8"))
        meta.update({key: bundle[key] for key in
                     ("weights_file", "tensor_state", "tensor_metadata")})

    if sklearn_payload is not None:
        with open(path.with_suffix(".skops"), "wb") as f:
            pickle.dump(sklearn_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta["sklearn_file"] = str(path.with_suffix(".skops").name)

    with open(_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def load_artifact(
    path: Union[str, Path],
    *,
    map_location: Any = None,
    allow_pickle: bool = False,
) -> Dict[str, Any]:
    """
    Load saved artifact components.

    Returns dict with keys ``state_dict``, ``sklearn_payload``, ``metadata``.
    """
    from corerec.api.model_bundle import is_safe_bundle, load_bundle
    sidecar = _meta_path(Path(path))
    has_python_payload = (sidecar.is_file() and
                          "sklearn_file" in json.loads(sidecar.read_text(encoding="utf-8")))
    if is_safe_bundle(path) and not has_python_payload:
        bundle = load_bundle(path, map_location=map_location, allow_pickle=allow_pickle)
        return {"metadata": bundle["config"], "state_dict": bundle["state_dict"],
                "sklearn_payload": None}
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
        from corerec.api.model_bundle import _load_bundle_components
        result["state_dict"] = _load_bundle_components(
            path, metadata, map_location, allow_pickle)["state_dict"]

    skops_name = metadata.get("sklearn_file")
    if skops_name:
        from corerec.api.model_bundle import _bundle_component, require_legacy_pickle
        skops_path = _bundle_component(path, skops_name)
        require_legacy_pickle(skops_path, allow_pickle)
        with open(skops_path, "rb") as f:
            result["sklearn_payload"] = pickle.load(f)

    return result
