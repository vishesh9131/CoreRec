"""
Reusable building blocks for recommender models.

These are the blocks CoreRec's own models are built from (HSTU, SASRec, DCN),
exposed so a new model can be assembled from tested parts:

    from corerec.nn import SASRecBlock, causal_mask

    x = item_emb(history) + pos_emb(positions)        # [B, n, d]
    for block in blocks:
        x = block(x, causal_mask(n, x.device), None)
"""

from typing import List, Optional

import torch
import torch.nn as nn

from corerec.engines.hstu import _HSTUBlock as HSTUBlock
from corerec.engines.hstu import _SASRecBlock as SASRecBlock


def causal_mask(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """[n, n] lower-triangular ones: position i may attend to positions <= i.

    The mask HSTUBlock and SASRecBlock take as their second argument.
    """
    return torch.tril(torch.ones(n, n, device=device))


class CrossLayer(nn.Module):
    """One DCN cross layer: x0 * (x . w) + b + x  (Wang et al., 2017)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xw = (x * self.weight).sum(dim=1, keepdim=True)  # [batch, 1]
        return x0 * xw + self.bias + x


class FMInteraction(nn.Module):
    """Second-order factorization-machine term over field embeddings.

    [B, F, d] -> [B]: sum over field pairs of <e_i, e_j>, in O(F d).
    """

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        square_of_sum = emb.sum(dim=1) ** 2
        sum_of_square = (emb ** 2).sum(dim=1)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1)


class MLP(nn.Sequential):
    """Linear -> ReLU -> Dropout, repeated; the last layer has no activation."""

    def __init__(self, dims: List[int], dropout: float = 0.0):
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        super().__init__(*layers)


__all__ = ["HSTUBlock", "SASRecBlock", "CrossLayer", "FMInteraction", "MLP", "causal_mask"]
