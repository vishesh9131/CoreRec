"""
Reference modules for corerec.nn.Recommender.

Small on purpose: each is a template to copy when writing a new model. Both
follow the contract in corerec.nn.recommender -- forward(query, items) ->
scores [B, K] -- and add score_all() so recommend() is one matmul.
"""

import torch
import torch.nn as nn

from corerec.nn.layers import HSTUBlock, SASRecBlock, causal_mask


class MatrixFactorization(nn.Module):
    """score(u, i) = <p_u, q_i> + b_i.  Use with inputs="user"."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.users = nn.Embedding(n_users, dim)
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        p = self.users(users).unsqueeze(1)                      # [B, 1, d]
        return (p * self.items(items)).sum(-1) + self.item_bias(items).squeeze(-1)

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        return self.users(users) @ self.items.weight.t() + self.item_bias.weight.t()


class SequentialTransformer(nn.Module):
    """Causal self-attention over the history (SASRec-style).  Use with inputs="history".

    Swap SASRecBlock for HSTUBlock, or write a new block, to try a different encoder.
    """

    def __init__(self, n_items: int, max_len: int, dim: int = 64, num_blocks: int = 2,
                 heads: int = 1, dropout: float = 0.2):
        super().__init__()
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(SASRecBlock(dim, heads, dropout) for _ in range(num_blocks))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """[B, L] left-padded history -> [B, d], the state after the newest item."""
        n = history.shape[1]
        x = self.items(history) + self.pos.weight[:n]
        x = self.dropout(x) * (history > 0).unsqueeze(-1)
        mask = causal_mask(n, history.device)
        for block in self.blocks:
            x = block(x, mask, None) * (history > 0).unsqueeze(-1)
        return self.norm(x[:, -1])

    def forward(self, history: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        h = self.encode(history).unsqueeze(1)                   # [B, 1, d]
        return (h * self.items(items)).sum(-1)

    def score_all(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode(history) @ self.items.weight.t()


class HSTUTransformer(nn.Module):
    """Causal HSTU blocks over the history (Zhai et al., 2024).  Use with inputs="history".

    SequentialTransformer with HSTUBlock in place of SASRecBlock: pointwise SiLU
    attention and a learned relative-position bias, so there's no position table.
    """

    def __init__(self, n_items: int, max_len: int, dim: int = 64, num_blocks: int = 2,
                 heads: int = 1, dropout: float = 0.2):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        # use_time=False: Recommender doesn't pass timestamps to the module
        self.blocks = nn.ModuleList(
            HSTUBlock(dim, heads, dim // heads, dim // heads, dropout, max_len, 1, False)
            for _ in range(num_blocks))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """[B, L] left-padded history -> [B, d], the state after the newest item."""
        valid = (history > 0).unsqueeze(-1)
        x = self.dropout(self.items(history)) * valid
        mask = causal_mask(history.shape[1], history.device)
        for block in self.blocks:
            x = block(x, mask, None) * valid
        return self.norm(x[:, -1])

    def forward(self, history: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        h = self.encode(history).unsqueeze(1)                   # [B, 1, d]
        return (h * self.items(items)).sum(-1)

    def score_all(self, history: torch.Tensor) -> torch.Tensor:
        return self.encode(history) @ self.items.weight.t()


__all__ = ["MatrixFactorization", "SequentialTransformer", "HSTUTransformer"]
