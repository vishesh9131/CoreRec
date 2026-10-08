"""
HSTU: generative next-item recommendation.

HSTU (Hierarchical Sequential Transduction Unit) is the encoder from Zhai et
al., "Actions Speak Louder than Words: Trillion-Parameter Sequential
Transducers for Generative Recommendations" (ICML 2024). It reads a user's
history as a sequence and is trained like a language model: at every position
it predicts the next item. Compared with SASRec it

- gates attention with SiLU instead of softmax, so the weights are not forced
  to sum to one (a strong signal from many past items is not diluted);
- multiplies the attention output by a learned gate ``U`` from the same input;
- adds learned biases for the distance between two positions and, when
  timestamps are given, for the time between the two interactions.

Training follows the paper's public ML-1M recipe: sampled softmax over 128
random negatives with temperature 0.05 and L2-normalised embeddings. The same
recipe trains a SASRec encoder when ``encoder="sasrec"``, which is how
``Findings/bench/generative_bench.py`` separates what the architecture adds
from what the loss adds.

Example
-------
    from corerec.engines import HSTU

    model = HSTU(epochs=20)
    model.fit(user_ids, item_ids, timestamps=timestamps)
    model.recommend(user_id, top_k=10)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from corerec.device import resolve_device
import torch.nn as nn
import torch.nn.functional as F

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ============================================================================
# Encoders
# ============================================================================

class _RelativeBias(nn.Module):
    """Learned attention bias for position distance and, optionally, time gap."""

    def __init__(self, max_len: int, num_time_buckets: int, use_time: bool):
        super().__init__()
        self.pos_w = nn.Parameter(torch.empty(max_len).normal_(0.0, 0.02))
        self.num_time_buckets = num_time_buckets
        self.ts_w = nn.Parameter(torch.empty(num_time_buckets + 1).normal_(0.0, 0.02)) if use_time else None

    def forward(self, n: int, ts: Optional[torch.Tensor]) -> torch.Tensor:
        idx = torch.arange(n, device=self.pos_w.device)
        bias = self.pos_w[(idx[:, None] - idx[None, :]).clamp(min=0)]  # [n, n]
        if self.ts_w is not None and ts is not None:
            gap = (ts[:, :, None] - ts[:, None, :]).abs().clamp(min=1.0)
            # log2-spaced buckets, as in the reference implementation
            buckets = (torch.log(gap) / 0.301).long().clamp(0, self.num_time_buckets)
            bias = bias + self.ts_w[buckets]  # [B, n, n]
        return bias


class _HSTUBlock(nn.Module):
    def __init__(self, d: int, heads: int, dqk: int, dv: int, dropout: float,
                 max_len: int, num_time_buckets: int, use_time: bool):
        super().__init__()
        self.d, self.heads, self.dqk, self.dv = d, heads, dqk, dv
        self.uvqk = nn.Parameter(torch.empty(d, heads * (2 * dv + 2 * dqk)).normal_(0.0, 0.02))
        self.o = nn.Linear(heads * dv, d)
        nn.init.xavier_uniform_(self.o.weight)
        self.bias = _RelativeBias(max_len, num_time_buckets, use_time)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, causal: torch.Tensor, ts: Optional[torch.Tensor]) -> torch.Tensor:
        B, n, _ = x.shape
        h = F.silu(F.layer_norm(x, (self.d,), eps=_EPS) @ self.uvqk)
        hv, hq = self.heads * self.dv, self.heads * self.dqk
        u, v, q, k = torch.split(h, [hv, hv, hq, hq], dim=-1)
        att = torch.einsum("bnhd,bmhd->bhnm", q.view(B, n, self.heads, self.dqk), k.view(B, n, self.heads, self.dqk))
        bias = self.bias(n, ts)
        att = att + (bias.unsqueeze(1) if bias.dim() == 3 else bias)
        # SiLU instead of softmax, scaled by the (fixed) sequence length
        att = F.silu(att) / n * causal
        out = torch.einsum("bhnm,bmhd->bnhd", att, v.view(B, n, self.heads, self.dv)).reshape(B, n, hv)
        out = u * F.layer_norm(out, (hv,), eps=_EPS)
        return x + self.o(F.dropout(out, p=self.dropout, training=self.training))


class _SASRecBlock(nn.Module):
    """The SASRec block (Kang & McAuley, 2018) as configured in the HSTU paper's baseline."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.attn_ln = nn.LayerNorm(d, eps=1e-8)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(d, eps=1e-8)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: torch.Tensor, ts: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.attn_ln(x)
        a, _ = self.attn(q, x, x, attn_mask=causal == 0, need_weights=False)
        x = q + a
        x = self.ffn_ln(x)
        return x + self.ffn(x)


class _SequenceNet(nn.Module):
    """Item + position embeddings, a stack of blocks, L2-normalised outputs."""

    def __init__(self, n_items: int, d: int, num_blocks: int, heads: int, max_len: int,
                 dropout: float, encoder: str, use_time: bool, num_time_buckets: int = 128):
        super().__init__()
        self.d, self.max_len, self.encoder = d, max_len, encoder
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        nn.init.normal_(self.item_emb.weight, 0.0, 0.02)
        nn.init.xavier_normal_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.dropout = nn.Dropout(dropout)
        if encoder == "hstu":
            self.blocks = nn.ModuleList(
                _HSTUBlock(d, heads, d // heads, d // heads, dropout, max_len, num_time_buckets, use_time)
                for _ in range(num_blocks))
            self.final_ln = None
        elif encoder == "sasrec":
            self.blocks = nn.ModuleList(_SASRecBlock(d, heads, dropout) for _ in range(num_blocks))
            self.final_ln = nn.LayerNorm(d, eps=1e-8)
        else:
            raise ValueError(f"encoder must be 'hstu' or 'sasrec', got {encoder!r}")
        self.register_buffer("causal", torch.tril(torch.ones(max_len, max_len)), persistent=False)

    def encode(self, ids: torch.Tensor, ts: Optional[torch.Tensor]) -> torch.Tensor:
        """ids: [B, n] right-padded with 0 (oldest first). Returns [B, n, d], unit length."""
        n = ids.size(1)
        valid = (ids > 0).unsqueeze(-1).float()
        x = self.item_emb(ids) * math.sqrt(self.d) + self.pos_emb.weight[:n]
        x = self.dropout(x) * valid
        causal = self.causal[:n, :n]
        for block in self.blocks:
            x = block(x, causal, ts) * valid
        if self.final_ln is not None:
            x = self.final_ln(x)
        return F.normalize(x, dim=-1, eps=_EPS)

    def item_vectors(self) -> torch.Tensor:
        return F.normalize(self.item_emb.weight, dim=-1, eps=_EPS)


# ============================================================================
# Recommender
# ============================================================================

class HSTU(BaseRecommender):
    """Generative sequential recommender (HSTU), trained on next-item prediction.

    Parameters follow the HSTU paper's ML-1M configuration, scaled to CPU-sized
    defaults where noted.

    Args:
        embedding_dim: Item and hidden size (paper: 50).
        num_blocks: Number of stacked layers (paper: 2; "HSTU-large": 8).
        num_heads: Attention heads; ``embedding_dim`` must divide by it.
        max_seq_length: Most recent interactions read per user (paper: 200).
        dropout: Dropout on embeddings and layer outputs (paper: 0.2).
        num_negatives: Random negatives per position in the sampled softmax (paper: 128).
        temperature: Softmax temperature on cosine scores (paper: 0.05).
        learning_rate, weight_decay, batch_size: Adam settings (paper: 1e-3, 0, 128).
        epochs: Passes over the training sequences (paper: 101).
        use_time: Add the time-gap attention bias when timestamps are given.
        encoder: ``"hstu"`` (default) or ``"sasrec"`` to train a SASRec encoder
            with the identical recipe, for like-for-like comparisons.
        seed: Seed for initialisation and negative sampling.
        device: Torch device; "auto" (default) picks CUDA, then Apple MPS, then CPU.
    """

    def __init__(
        self,
        embedding_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        max_seq_length: int = 200,
        dropout: float = 0.2,
        num_negatives: int = 128,
        temperature: float = 0.05,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        epochs: int = 100,
        use_time: bool = True,
        encoder: str = "hstu",
        seed: Optional[int] = 42,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False,
        name: str = "HSTU",
    ):
        super().__init__(name=name, verbose=verbose)
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if encoder not in ("hstu", "sasrec"):
            raise ValueError(f"encoder must be 'hstu' or 'sasrec', got {encoder!r}")
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_time = use_time
        self.encoder = encoder
        self.seed = seed
        self.device = resolve_device(device)

        self.model: Optional[_SequenceNet] = None
        self.item_to_index: Dict[Any, int] = {}
        self.index_to_item: List[Any] = [None]          # index 0 is padding
        self.user_sequences: Dict[Any, np.ndarray] = {}  # user -> item indices, oldest first
        self.user_times: Dict[Any, np.ndarray] = {}
        self.has_time = False
        self.window = max_seq_length                    # padded length, fixed at fit time
        self.history: List[float] = []
        self.is_fitted = False

    # ------------------------------------------------------------------ fit

    def fit(self, user_ids: Sequence[Any] = None, item_ids: Sequence[Any] = None,
            ratings: Optional[Sequence[float]] = None, timestamps: Optional[Sequence[float]] = None,
            on_epoch_end: Optional[Any] = None, **kwargs: Any) -> "HSTU":
        """Fit on one row per interaction.

        Each user's interactions are ordered by ``timestamps`` when given, and
        otherwise kept in the order they arrive. ``ratings`` is accepted for
        API compatibility; every listed interaction counts as a positive.
        ``on_epoch_end(epoch, model)``, if given, is called after each epoch.
        """
        if user_ids is None or item_ids is None:
            raise TypeError("HSTU.fit() needs user_ids and item_ids (one row per interaction)")
        users = list(user_ids)
        items = list(item_ids)
        if len(users) != len(items) or not users:
            raise ValueError("user_ids and item_ids must be non-empty and the same length")
        times = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        if times is not None and len(times) != len(users):
            raise ValueError("timestamps must have one entry per interaction")

        if self.seed is not None:
            torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.item_to_index = {}
        self.index_to_item = [None]
        for it in items:
            if it not in self.item_to_index:
                self.item_to_index[it] = len(self.index_to_item)
                self.index_to_item.append(it)
        n_items = len(self.index_to_item) - 1

        order = np.arange(len(users)) if times is None else np.argsort(times, kind="stable")
        seqs: Dict[Any, List[int]] = {}
        tss: Dict[Any, List[float]] = {}
        for r in order:
            u = users[r]
            seqs.setdefault(u, []).append(self.item_to_index[items[r]])
            tss.setdefault(u, []).append(0.0 if times is None else float(times[r]))
        self.user_sequences = {u: np.asarray(s, dtype=np.int64) for u, s in seqs.items()}
        self.user_times = {u: np.asarray(t, dtype=np.float64) for u, t in tss.items()}
        self.has_time = times is not None and self.use_time

        longest = max(len(s) for s in self.user_sequences.values())
        self.window = int(min(self.max_seq_length, max(longest - 1, 1)))
        self.model = _SequenceNet(n_items, self.embedding_dim, self.num_blocks, self.num_heads, self.window,
                                  self.dropout, self.encoder, self.has_time).to(self.device)

        X, Y, T = self._training_windows()
        if len(X) == 0:
            raise ValueError("HSTU needs at least one user with two or more interactions")
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.history = []
        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(len(X))
            total, count = 0.0, 0
            for start in range(0, len(perm), self.batch_size):
                b = perm[start:start + self.batch_size]
                loss = self._sampled_softmax_loss(X[b], Y[b], None if T is None else T[b], n_items, rng)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
                count += len(b)
            self.history.append(total / max(count, 1))
            if self.verbose:
                logger.info("%s epoch %d/%d loss %.4f", self.name, epoch + 1, self.epochs, self.history[-1])
            if on_epoch_end is not None:
                self.model.eval()
                self.is_fitted = True
                on_epoch_end(epoch + 1, self)
        self.model.eval()
        self.is_fitted = True
        return self

    def _training_windows(self):
        """Per user: the latest window of (input, next-item target) pairs, right-padded."""
        n = self.window
        X, Y, T = [], [], []
        for u, seq in self.user_sequences.items():
            if len(seq) < 2:
                continue
            x, y, t = seq[:-1][-n:], seq[1:][-n:], self.user_times[u][:-1][-n:]
            pad = n - len(x)
            X.append(np.pad(x, (0, pad)))
            Y.append(np.pad(y, (0, pad)))
            T.append(np.pad(t, (0, pad), mode="edge" if len(t) else "constant"))
        X, Y = np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)
        T = np.asarray(T, dtype=np.float32) if self.has_time else None
        return X, Y, T

    def _sampled_softmax_loss(self, x, y, t, n_items, rng) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device)
        y = torch.as_tensor(y, device=self.device)
        ts = None if t is None else torch.as_tensor(t, device=self.device)
        h = self.model.encode(x, ts)                        # [B, n, d]
        items = self.model.item_vectors()
        # One set of random negatives per sequence, shared by its positions. The
        # reference draws a fresh set per position; here each position still sees
        # K uniform negatives, only correlated across positions, at a sixth of the
        # CPU cost.
        neg_idx = torch.as_tensor(rng.integers(1, n_items + 1, size=(len(y), self.num_negatives)),
                                  device=self.device)                      # [B, K]
        neg = torch.einsum("bnd,bkd->bnk", h, items[neg_idx])                # [B, n, K]
        neg = neg.masked_fill(neg_idx[:, None, :] == y[:, :, None], float("-inf"))
        mask = y > 0
        pos = (h[mask] * items[y[mask]]).sum(-1, keepdim=True)              # [M, 1]
        neg = neg[mask]                                                     # [M, K]
        y = y[mask]
        logits = torch.cat([pos, neg], dim=1) / self.temperature
        return F.cross_entropy(logits, torch.zeros(len(y), dtype=torch.long, device=self.device))

    # ------------------------------------------------------------ inference

    def _user_vectors(self, users: Sequence[Any]) -> torch.Tensor:
        """Encode each user's latest window; returns [len(users), d] on the model's device."""
        n = self.window
        X = np.zeros((len(users), n), dtype=np.int64)
        T = np.zeros((len(users), n), dtype=np.float32)
        last = np.zeros(len(users), dtype=np.int64)
        for r, u in enumerate(users):
            seq, ts = self.user_sequences[u][-n:], self.user_times[u][-n:]
            X[r, :len(seq)] = seq
            T[r, :len(seq)] = ts
            if len(ts) < n:
                T[r, len(seq):] = ts[-1]
            last[r] = len(seq) - 1
        self.model.eval()
        with torch.no_grad():
            h = self.model.encode(torch.as_tensor(X, device=self.device),
                                  torch.as_tensor(T, device=self.device) if self.has_time else None)
        return h[torch.arange(len(users), device=self.device), torch.as_tensor(last, device=self.device)]

    @torch.no_grad()
    def score_users(self, users: Sequence[Any], batch_size: int = 256) -> np.ndarray:
        """Scores of every item for each user, [len(users), n_items], in index order 1..n."""
        self._check_fitted()
        self.model.eval()
        items = self.model.item_vectors()[1:]
        out = []
        for s in range(0, len(users), batch_size):
            out.append((self._user_vectors(users[s:s + batch_size]) @ items.T).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.index_to_item) - 1))

    def recommend(self, user_id: Any, top_k: int = 10, exclude_seen: bool = True,
                  exclude_items: Optional[Sequence[Any]] = None, **kwargs: Any) -> List[Any]:
        self._check_fitted()
        if user_id not in self.user_sequences:
            return []
        scores = self.score_users([user_id])[0]
        if exclude_seen:
            scores[self.user_sequences[user_id] - 1] = -np.inf
        for it in exclude_items or ():
            idx = self.item_to_index.get(it)
            if idx is not None:
                scores[idx - 1] = -np.inf
        k = min(top_k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [self.index_to_item[i + 1] for i in top]

    def predict(self, user_id: Any, item_id: Any, **kwargs: Any) -> float:
        return self.batch_predict([(user_id, item_id)])[0]

    @torch.no_grad()
    def batch_predict(self, pairs: List[tuple], **kwargs: Any) -> List[float]:
        self._check_fitted()
        users = [u for u, _ in pairs if u in self.user_sequences]
        uniq = list(dict.fromkeys(users))
        vecs = {}
        if uniq:
            for u, v in zip(uniq, self._user_vectors(uniq)):
                vecs[u] = v
        items = self.model.item_vectors()
        out = []
        for u, it in pairs:
            idx = self.item_to_index.get(it)
            out.append(0.0 if u not in vecs or idx is None else float(vecs[u] @ items[idx]))
        return out

    # ---------------------------------------------------------- persistence

    def _config(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim, "num_blocks": self.num_blocks, "num_heads": self.num_heads,
            "max_seq_length": self.max_seq_length, "dropout": self.dropout, "num_negatives": self.num_negatives,
            "temperature": self.temperature, "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay, "batch_size": self.batch_size, "epochs": self.epochs,
            "use_time": self.use_time, "encoder": self.encoder, "seed": self.seed, "verbose": self.verbose,
            "name": self.name,
        }

    def save(self, path: Union[str, Path], **kwargs: Any) -> None:
        from corerec.api.torch_bundle import save_torch_production

        self._check_fitted()
        users = list(self.user_sequences)
        lengths = np.asarray([len(self.user_sequences[u]) for u in users], dtype=np.int64)
        arrays = {
            "seq_items": np.concatenate([self.user_sequences[u] for u in users]),
            "seq_times": np.concatenate([self.user_times[u] for u in users]),
            "seq_lengths": lengths,
        }
        state = {
            "users": users, "items": self.index_to_item[1:], "has_time": self.has_time,
            "window": self.window, "is_fitted": True,
        }
        save_torch_production(self, path, config=self._config(), state=state, arrays=arrays)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[Union[str, torch.device]] = None) -> "HSTU":
        from corerec.api.torch_bundle import load_torch_production

        def _factory(cfg):
            return cls(device=device, **cfg)

        def _restore(inst, cfg, state, arrays, bundle):
            inst.index_to_item = [None] + list(state["items"])
            inst.item_to_index = {it: i for i, it in enumerate(inst.index_to_item) if i}
            inst.has_time = bool(state["has_time"])
            inst.window = int(state["window"])
            bounds = np.concatenate([[0], np.cumsum(arrays["seq_lengths"])])
            inst.user_sequences, inst.user_times = {}, {}
            for r, u in enumerate(state["users"]):
                a, b = bounds[r], bounds[r + 1]
                inst.user_sequences[u] = arrays["seq_items"][a:b].astype(np.int64)
                inst.user_times[u] = arrays["seq_times"][a:b].astype(np.float64)
            inst.is_fitted = True

        def _build(inst, bundle):
            inst.model = _SequenceNet(len(inst.index_to_item) - 1, inst.embedding_dim, inst.num_blocks,
                                      inst.num_heads, inst.window, inst.dropout, inst.encoder,
                                      inst.has_time).to(inst.device)

        loaded = load_torch_production(cls, path, build_model=_build, factory=_factory, restore=_restore,
                                       map_location=device)
        if loaded is None:
            raise FileNotFoundError(f"No HSTU bundle at {path}")
        return loaded
