"""
Pick the torch device a model trains and predicts on.

    resolve_device()          # "auto": CUDA, then Apple MPS, then CPU
    resolve_device("mps")     # explicit; falls back to CPU with a warning
                              # when this machine doesn't have it

The fallback matters for loading: a model trained on a Mac (mps) or a GPU box
(cuda) records that device, and has to load on a CPU-only server.
"""

import warnings
from typing import Optional, Union

import torch


def mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def resolve_device(device: Optional[Union[str, torch.device]] = "auto",
                   needs_sparse: bool = False) -> torch.device:
    """needs_sparse: the model uses torch.sparse, which MPS can't run (torch 2.2)."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available() and not needs_sparse:
            return torch.device("mps")
        return torch.device("cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(f"device={str(device)!r} requested but CUDA isn't available; using CPU")
        return torch.device("cpu")
    if dev.type == "mps" and not mps_available():
        warnings.warn(f"device={str(device)!r} requested but MPS isn't available; using CPU")
        return torch.device("cpu")
    if dev.type == "mps" and needs_sparse:
        warnings.warn("this model needs sparse tensors, which MPS doesn't support; using CPU")
        return torch.device("cpu")
    return dev
