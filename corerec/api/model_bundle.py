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
    import os
    import tempfile
    import uuid

    base = artifact_base(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "format": SAFE_FORMAT,
        "corerec_save_version": COREREC_SAVE_VERSION,
        "model_class": model_class,
        "config": _jsonify(config),
        "state": _jsonify(state),
    }

    # Publish the metadata pointer last; replacing components in place could
    # corrupt the previous bundle if a later replacement fails.
    generation = uuid.uuid4().hex
    staged = []  # (temporary, final)
    published = []
    committed = False

    def stage(final: Path, write) -> None:
        with tempfile.NamedTemporaryFile(dir=base.parent, prefix=f".{final.name}.",
                                         delete=False) as f:
            staged.append((Path(f.name), final))
            write(f)
            f.flush()
            os.fsync(f.fileno())

    try:
        if state_dict is not None:
            try:
                import torch
            except ImportError as e:
                raise SaveLoadError("torch required to save state_dict bundle") from e
            weights_path = base.with_name(f"{base.name}.{generation}.weights.pt")
            stage(weights_path, lambda f: torch.save(state_dict, f))
            meta["weights_file"] = weights_path.name

        if arrays:
            try:
                import numpy as np
            except ImportError as e:
                raise SaveLoadError("numpy required to save array bundle") from e
            npz_path = base.with_name(f"{base.name}.{generation}.arrays.npz")
            stage(npz_path, lambda f: np.savez_compressed(f, **arrays))
            meta["arrays_file"] = npz_path.name

        meta_path = base.with_suffix(".meta.json")
        text = json.dumps(meta, indent=2, default=str).encode("utf-8")
        stage(meta_path, lambda f: f.write(text))
        for temporary, final in staged:
            os.replace(temporary, final)
            published.append(final)
        committed = True
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        if not committed:
            for final in published:
                final.unlink(missing_ok=True)
    # shortcut: old generations stay readable for concurrent loaders; add
    # explicit artifact cleanup when frequent checkpointing needs reclamation.
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


def pack_arrays(values: Dict[str, Any]):
    """Split ndarrays and scipy sparse matrices into npz-safe arrays.

    Returns ``(arrays, sparse_names)``: a sparse ``X`` becomes ``X.data``,
    ``X.indices``, ``X.indptr`` and ``X.shape`` (CSR), so loading needs no pickle.
    """
    import numpy as np

    arrays, sparse = {}, []
    for name, v in values.items():
        if hasattr(v, "tocsr"):
            m = v.tocsr()
            arrays.update({f"{name}.data": m.data, f"{name}.indices": m.indices,
                           f"{name}.indptr": m.indptr, f"{name}.shape": np.asarray(m.shape)})
            sparse.append(name)
        else:
            arrays[name] = np.asarray(v)
    return arrays, sparse


def unpack_arrays(arrays: Dict[str, Any], sparse_names) -> Dict[str, Any]:
    """Inverse of pack_arrays."""
    from scipy.sparse import csr_matrix

    out = {k: v for k, v in arrays.items() if "." not in k}
    for name in sparse_names:
        out[name] = csr_matrix((arrays[f"{name}.data"], arrays[f"{name}.indices"],
                                arrays[f"{name}.indptr"]), shape=tuple(arrays[f"{name}.shape"]))
    return out


def ordered_ids(id_map: Dict[Any, int]) -> list:
    """Ids of a ``{id: code}`` map in code order, for JSON."""
    return sorted(id_map, key=id_map.get)


def warn_legacy_pickle(path: Union[str, Path]) -> None:
    import warnings

    warnings.warn(f"{path} is a legacy pickle, which can run code when loaded. Load it only "
                  "if you trust it, then save() it again to convert it to the safe format.",
                  DeprecationWarning, stacklevel=3)


def save_legacy_pickle(path: Union[str, Path], payload: Any) -> None:
    """Explicit opt-in legacy pickle (discouraged in production)."""
    import warnings

    warnings.warn(
        "Saving with pickle is deprecated for production. Use safe=True (default).",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(p, payload)


def atomic_pickle_dump(path: Union[str, Path], payload: Any) -> None:
    """Replace a pickle artifact only after its new contents are fully written."""
    import os
    import pickle
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent, prefix=f".{p.name}.", delete=False) as f:
            temporary = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, p)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
