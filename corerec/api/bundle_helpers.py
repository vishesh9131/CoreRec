"""Shared helpers for safe model bundle save/load."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse


def coerce_id(value: Any) -> Any:
    """Restore integer IDs serialized as JSON strings (e.g. ``\"0\"`` → ``0``)."""
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


def pairs(mapping: Optional[Dict[Any, Any]]) -> List[List[Any]]:
    if not mapping:
        return []
    return [[k, v] for k, v in mapping.items()]


def dict_from_pairs(
    pairs_list: Optional[List[List[Any]]],
    *,
    int_keys: bool = False,
    coerce_numeric: bool = True,
) -> Dict[Any, Any]:
    if not pairs_list:
        return {}
    out: Dict[Any, Any] = {}
    for entry in pairs_list:
        key = entry[0]
        val = entry[1]
        if int_keys or coerce_numeric:
            key = coerce_id(key)
        if coerce_numeric:
            val = coerce_id(val)
        out[key] = val
    return out


def nested_lists(mapping: Optional[Dict[Any, Any]]) -> List[List[Any]]:
    if not mapping:
        return []
    return [[k, list(v)] for k, v in mapping.items()]


def nested_dict_from_lists(pairs_list: Optional[List[List[Any]]]) -> Dict[Any, Any]:
    if not pairs_list:
        return {}
    return {k: list(v) for k, v in pairs_list}


def tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def sparse_to_dense(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if sparse.issparse(value):
        return value.toarray()
    return np.asarray(value)


def dense_to_sparse_csr(value: Any):
    if value is None:
        return None
    if sparse.issparse(value):
        return value.tocsr()
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return sparse.csr_matrix(arr)


def save_map_state(**maps: Optional[Dict[Any, Any]]) -> Dict[str, List[List[Any]]]:
    return {f"{name}_pairs": pairs(data) for name, data in maps.items()}


def load_map_state(
    state: Dict[str, Any],
    *names: str,
    int_key_names: Optional[tuple] = None,
    coerce_numeric: bool = True,
) -> Dict[str, Dict[Any, Any]]:
    int_key_names = int_key_names or ()
    loaded: Dict[str, Dict[Any, Any]] = {}
    for name in names:
        loaded[name] = dict_from_pairs(
            state.get(f"{name}_pairs"),
            int_keys=name in int_key_names,
            coerce_numeric=coerce_numeric,
        )
    return loaded


def save_feature_map(feature_map: Optional[Dict[str, Dict[Any, Any]]]) -> Dict[str, List[List[Any]]]:
    if not feature_map:
        return {"feature_map_entries": []}
    return {"feature_map_entries": [[k, pairs(v)] for k, v in feature_map.items()]}


def load_feature_map(state: Dict[str, Any]) -> Dict[str, Dict[Any, Any]]:
    entries = state.get("feature_map_entries")
    if entries:
        return {k: dict_from_pairs(v, coerce_numeric=True) for k, v in entries}
    legacy = state.get("feature_map")
    if isinstance(legacy, dict):
        return legacy
    return {}
