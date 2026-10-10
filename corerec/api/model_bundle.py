"""
Safe model bundle persistence (production default).

Format ``corerec_safe_v1``::

    {base}.meta.json     — config + JSON-safe state (no pickle)
    {base}.{generation}.weights.npz — tensor bytes and JSON dtype/shape metadata
    {base}.{generation}.arrays.npz — numpy arrays (compressed)

Legacy checkpoints remain loadable only with explicit ``allow_pickle=True``.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from corerec.api.exceptions import SaveLoadError

COREREC_SAVE_VERSION = "1.0"
SAFE_FORMAT = "corerec_safe_v1"


def artifact_base(path: Union[str, Path]) -> Path:
    """Normalize ``/tmp/model.pt`` → ``/tmp/model``."""
    p = Path(path)
    for ext in (".weights.pt", ".weights.npz", ".arrays.npz", ".meta.json", ".pt", ".pkl", ".npy"):
        if p.name.endswith(ext):
            name = p.name[: -len(ext)]
            stem, _, token = name.rpartition(".")
            if ext in (".weights.pt", ".weights.npz", ".arrays.npz") and len(token) == 32 and all(
                c in "0123456789abcdef" for c in token
            ):
                name = stem
            return p.with_name(name)
    return p


def bundle_meta_path(path: Union[str, Path]) -> Path:
    base = artifact_base(path)
    current = base.with_name(base.name + ".meta.json")
    legacy = base.with_suffix(".meta.json")
    return legacy if not current.exists() and legacy.exists() else current


def require_legacy_pickle(path: Union[str, Path], allow_pickle: bool = False) -> None:
    """Reject legacy deserialization unless the caller explicitly trusts the file."""
    import warnings

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    if not allow_pickle:
        raise SaveLoadError("Legacy loading can execute arbitrary code. Only for a trusted "
                            "artifact, pass allow_pickle=True; re-save it as a safe bundle.")
    warnings.warn("Loading a legacy artifact can execute arbitrary code; only load trusted files.",
                  UserWarning, stacklevel=3)


@contextmanager
def _atomic_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bundle_component(base: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise SaveLoadError("Bundle component must be a filename in the artifact directory")
    component = base.parent / name
    if component.resolve().parent != base.parent.resolve():
        raise SaveLoadError("Bundle component must remain in the artifact directory")
    return component


def is_safe_bundle(path: Union[str, Path]) -> bool:
    meta = bundle_meta_path(path)
    if not meta.is_file():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("format") == SAFE_FORMAT
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

    # New components remain invisible to readers until the metadata is replaced.
    generation = uuid.uuid4().hex
    old_meta = {}
    meta_path = base.with_name(base.name + ".meta.json")
    try:
        previous = json.loads(bundle_meta_path(path).read_text(encoding="utf-8"))
        old_meta = previous if isinstance(previous, dict) else {}
    except (OSError, ValueError):
        pass
    written = []
    try:
        if state_dict is not None:
            import numpy as np
            import torch
            tensors, descriptions = {}, {}
            for index, (name, tensor) in enumerate(state_dict.items()):
                if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                    raise SaveLoadError("Safe weights must be dense tensors")
                key = f"tensor_{index}"
                tensors[key] = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                descriptions[name] = {"array": key, "dtype": str(tensor.dtype).removeprefix("torch."),
                                      "shape": list(tensor.shape)}
            weights_path = base.with_name(f"{base.name}.{generation}.weights.npz")
            written.append(weights_path)
            with weights_path.open("wb") as f:
                np.savez_compressed(f, **tensors)
                f.flush()
                os.fsync(f.fileno())
            meta["weights_file"] = weights_path.name
            meta["tensor_state"] = descriptions
            meta["tensor_metadata"] = _jsonify(getattr(state_dict, "_metadata", {}))
        if arrays:
            import numpy as np
            if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
                raise SaveLoadError("Safe bundle arrays cannot contain Python objects")
            npz_path = base.with_name(f"{base.name}.{generation}.arrays.npz")
            written.append(npz_path)
            with npz_path.open("wb") as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            meta["arrays_file"] = npz_path.name
        with _atomic_file(meta_path) as f:
            f.write(json.dumps(meta, indent=2, default=str).encode("utf-8"))
    except BaseException:
        for component in written:
            component.unlink(missing_ok=True)
        raise
    # Remove only files from a previous generation created by this writer.
    for key, suffix in (("weights_file", ".weights.pt"), ("weights_file", ".weights.npz"),
                        ("arrays_file", ".arrays.npz")):
        name = old_meta.get(key, "")
        prefix = base.name + "."
        token = name[len(prefix):-len(suffix)] if isinstance(name, str) else ""
        if (isinstance(name, str) and name.startswith(prefix) and name.endswith(suffix)
                and len(token) == 32 and all(c in "0123456789abcdef" for c in token)):
            try:
                (base.parent / name).unlink(missing_ok=True)
            except OSError:
                pass
    return base


def load_bundle(path: Union[str, Path], *, map_location: Any = None, allow_pickle: bool = False) -> Dict[str, Any]:
    """Load a safe bundle, retrying if a concurrent save replaced its generation."""
    base = artifact_base(path)
    meta_path = bundle_meta_path(path)
    for _ in range(3):
        if not meta_path.is_file():
            raise SaveLoadError(f"Safe bundle metadata not found: {meta_path}")
        snapshot = meta_path.read_text(encoding="utf-8")
        meta = json.loads(snapshot)
        if not isinstance(meta, dict):
            raise SaveLoadError("Safe bundle metadata must be a JSON object")
        if meta.get("format") != SAFE_FORMAT:
            raise SaveLoadError(f"Unsupported bundle format: {meta.get('format')!r}")
        try:
            return _load_bundle_components(base, meta, map_location, allow_pickle)
        except FileNotFoundError:
            if meta_path.read_text(encoding="utf-8") == snapshot:
                raise
    raise SaveLoadError("Bundle changed repeatedly during loading; retry the load")


def _load_bundle_components(base: Path, meta: Dict[str, Any], map_location: Any, allow_pickle: bool) -> Dict[str, Any]:
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

        weights_path = _bundle_component(base, weights_name)
        if "tensor_state" in meta:
            import numpy as np
            dtypes = {str(dtype).removeprefix("torch."): dtype for dtype in (
                torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                torch.float16, torch.bfloat16, torch.float32, torch.float64,
                torch.complex64, torch.complex128,
            )}
            state_dict = OrderedDict()
            with np.load(weights_path, allow_pickle=False) as tensors:
                for name, description in meta["tensor_state"].items():
                    dtype = dtypes.get(description["dtype"])
                    if dtype is None:
                        raise SaveLoadError(f"Unsupported tensor dtype: {description['dtype']}")
                    raw = tensors[description["array"]]
                    if raw.dtype != np.uint8 or raw.ndim != 1:
                        raise SaveLoadError("Tensor storage must be a one-dimensional byte array")
                    tensor = torch.from_numpy(raw.copy()).view(dtype).reshape(description["shape"])
                    if map_location is not None and isinstance(map_location, (str, torch.device)):
                        tensor = tensor.to(map_location)
                    state_dict[name] = tensor
            state_dict._metadata = meta.get("tensor_metadata", {})
            result["state_dict"] = state_dict
        else:
            import re
            version = re.match(r"^(\d+)\.(\d+)", torch.__version__)
            if not version or tuple(map(int, version.groups())) < (2, 10):
                if not allow_pickle:
                    raise SaveLoadError("Older torch weights require PyTorch >=2.10 for restricted "
                                        "loading, or allow_pickle=True for a trusted artifact")
                require_legacy_pickle(weights_path, allow_pickle)
            result["state_dict"] = torch.load(
                weights_path, map_location=map_location, weights_only=True
            )

    arrays_name = meta.get("arrays_file")
    if arrays_name:
        import numpy as np

        npz_path = _bundle_component(base, arrays_name)
        with np.load(npz_path, allow_pickle=False) as npz:
            result["arrays"] = {k: npz[k] for k in npz.files}

    return result


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
    import pickle

    with _atomic_file(Path(path)) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
