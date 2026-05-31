"""Shared helpers for safe model bundle save/load."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse


def pairs(mapping: Optional[Dict[Any, Any]]) -> List[List[Any]]:
    if not mapping:
        return []
    return [[k, v] for k, v in mapping.items()]


def dict_from_pairs(
    pairs_list: Optional[List[List[Any]]],
    *,
    int_keys: bool = False,
) -> Dict[Any, Any]:
    if not pairs_list:
        return {}
    out: Dict[Any, Any] = {}
    for entry in pairs_list:
        key = int(entry[0]) if int_keys else entry[0]
        out[key] = entry[1]
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
) -> Dict[str, Dict[Any, Any]]:
    int_key_names = int_key_names or ()
    loaded: Dict[str, Dict[Any, Any]] = {}
    for name in names:
        loaded[name] = dict_from_pairs(
            state.get(f"{name}_pairs"),
            int_keys=name in int_key_names,
        )
    return loaded
