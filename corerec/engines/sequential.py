"""Production-grade sequential (session/next-item) recommenders.

A shared base (:class:`_SequentialBase`) provides the unified contract --
``fit`` (builds per-user chronological sequences, sliding-window next-item samples,
BCE with negative sampling), ``predict``, a vectorized ``recommend`` (scores the
whole catalogue from the user's recent history), ``save``/``load`` with exact
round-trip, a device flag, and a post-fit collapse guard.

Each concrete model implements only its sequence scorer. Encoder models
(GRU4Rec, Caser, BST) summarize the history into one vector and score by dot
product; target-aware models (DIN, DIEN) attend over the history with respect to
each candidate. Both expose the same ``forward_scores(seq, cand)`` interface so
the base trains and serves them uniformly.

Reference: Hidasi 2016 (GRU4Rec); Tang 2018 (Caser); Chen 2019 (BST);
Zhou 2018 (DIN); Zhou 2019 (DIEN).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sequence scorers. Item index 0 is reserved for padding.
# --------------------------------------------------------------------------- #
class _SeqNet(nn.Module):
    """Holds the item table + a pluggable scorer; index 0 == padding."""

    def __init__(self, n_items, dim, scorer):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        nn.init.zeros_(self.item_bias.weight)
        self.scorer = scorer
        self.dim = dim

    def forward_scores(self, seq, cand):
        """seq [B, L] (padded ids), cand [B, C] (candidate ids) -> [B, C]."""
        mask = (seq > 0).float()                       # [B, L]
        seq_emb = self.item_emb(seq)                   # [B, L, d]
        cand_emb = self.item_emb(cand)                 # [B, C, d]
        s = self.scorer(seq_emb, mask, cand_emb)       # [B, C]
        return s + self.item_bias(cand).squeeze(-1)


class _GRUScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, seq_emb, mask, cand_emb):
        out, _ = self.gru(seq_emb)
        last = (mask.sum(1).clamp(min=1).long() - 1)        # last non-pad index
        repr = out[torch.arange(out.size(0)), last]         # [B, d]
        return torch.bmm(cand_emb, repr.unsqueeze(-1)).squeeze(-1)


class _CaserScorer(nn.Module):
    def __init__(self, dim, L=50, nh=16, nv=4):
        super().__init__()
        self.hconvs = nn.ModuleList([nn.Conv2d(1, nh, (h, dim)) for h in (2, 3, 4)])
        self.vconv = nn.Conv2d(1, nv, (L, 1))
        self.fc = nn.Linear(nh * 3 + nv * dim, dim)
        self.L = L

    def forward(self, seq_emb, mask, cand_emb):
        B, L, d = seq_emb.shape
        if L < self.L:
            pad = seq_emb.new_zeros(B, self.L - L, d)
            x = torch.cat([pad, seq_emb], dim=1)
        else:
            x = seq_emb[:, -self.L:]
        x = x.unsqueeze(1)                                   # [B,1,L,d]
        h = [torch.relu(c(x)).max(dim=2)[0].squeeze(-1) for c in self.hconvs]
        v = torch.relu(self.vconv(x)).view(B, -1)
        repr = self.fc(torch.cat(h + [v], dim=1))            # [B, d]
        return torch.bmm(cand_emb, repr.unsqueeze(-1)).squeeze(-1)


class _BSTScorer(nn.Module):
    def __init__(self, dim, L=50, heads=2, layers=2, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(L + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.L = L

    def forward(self, seq_emb, mask, cand_emb):
        B, L, d = seq_emb.shape
        pos = torch.arange(1, L + 1, device=seq_emb.device).clamp(max=self.L)
        x = seq_emb + self.pos(pos).unsqueeze(0)
        key_pad = (mask == 0)
        h = self.enc(x, src_key_padding_mask=key_pad)
        last = (mask.sum(1).clamp(min=1).long() - 1)
        repr = h[torch.arange(B), last]
        return torch.bmm(cand_emb, repr.unsqueeze(-1)).squeeze(-1)


class _DINScorer(nn.Module):
    """Target-aware attention pooling of the history w.r.t. each candidate."""

    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.out = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, seq_emb, mask, cand_emb):
        B, L, d = seq_emb.shape
        C = cand_emb.size(1)
        h = seq_emb.unsqueeze(1).expand(B, C, L, d)                  # [B,C,L,d]
        c = cand_emb.unsqueeze(2).expand(B, C, L, d)                 # [B,C,L,d]
        feat = torch.cat([h, c, h - c, h * c], dim=-1)
        a = self.att(feat).squeeze(-1)                               # [B,C,L]
        a = a.masked_fill(mask.unsqueeze(1) == 0, -1e9).softmax(-1)
        pooled = (a.unsqueeze(-1) * h).sum(2)                        # [B,C,d]
        return self.out(torch.cat([pooled, cand_emb], dim=-1)).squeeze(-1)


class _DIENScorer(nn.Module):
    """DIN-style attention over GRU interest states (AUGRU approximated by an
    attention-weighted second pass)."""

    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.att = nn.Sequential(nn.Linear(dim * 4, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.out = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, seq_emb, mask, cand_emb):
        B, L, d = seq_emb.shape
        C = cand_emb.size(1)
        states, _ = self.gru(seq_emb)                               # [B,L,d] interest states
        h = states.unsqueeze(1).expand(B, C, L, d)
        c = cand_emb.unsqueeze(2).expand(B, C, L, d)
        feat = torch.cat([h, c, h - c, h * c], dim=-1)
        a = self.att(feat).squeeze(-1)
        a = a.masked_fill(mask.unsqueeze(1) == 0, -1e9).softmax(-1)
        interest = (a.unsqueeze(-1) * h).sum(2)                     # [B,C,d]
        return self.out(torch.cat([interest, cand_emb], dim=-1)).squeeze(-1)


class _NARMScorer(nn.Module):
    """Neural Attentive Recommendation Machine (Li 2017): GRU + attention over
    hidden states, global+local representation, bilinear scoring."""
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.A1 = nn.Linear(dim, dim, bias=False)
        self.A2 = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)
        self.proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, seq_emb, mask, cand_emb):
        out, _ = self.gru(seq_emb)                          # [B,L,d]
        last = (mask.sum(1).clamp(min=1).long() - 1)
        ht = out[torch.arange(out.size(0)), last]           # global [B,d]
        a = self.v(torch.sigmoid(self.A1(ht).unsqueeze(1) + self.A2(out)))  # [B,L,1]
        a = a.masked_fill(mask.unsqueeze(-1) == 0, -1e9).softmax(1)
        cl = (a * out).sum(1)                               # local [B,d]
        repr = self.proj(torch.cat([ht, cl], dim=1))        # [B,d]
        return torch.bmm(cand_emb, repr.unsqueeze(-1)).squeeze(-1)


_SCORERS = {
    "GRU4Rec": lambda d, L: _GRUScorer(d),
    "NARM": lambda d, L: _NARMScorer(d),
    "Caser": lambda d, L: _CaserScorer(d, L),
    "BST": lambda d, L: _BSTScorer(d, L),
    "DIN": lambda d, L: _DINScorer(d),
    "DIEN": lambda d, L: _DIENScorer(d),
}


class _SequentialBase(BaseRecommender):
    MODEL = "GRU4Rec"

    def __init__(self, name: str = None, embedding_dim: int = 64,
                 max_seq_len: int = 50, learning_rate: float = 1e-3,
                 batch_size: int = 1024, epochs: int = 20, num_negatives: int = 1,
                 reg: float = 1e-6, verbose: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42, trainable: bool = True):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_negatives = num_negatives
        self.reg = reg
        self.device = device
        self.seed = seed
        self.model = None
        self.user_seq = {}      # user_id -> list of item_idx (1..n_items), chronological

    def _make_scorer(self):
        return _SCORERS[self.MODEL](self.embedding_dim, self.max_seq_len)

    def _pad(self, seq):
        seq = seq[-self.max_seq_len:]
        return [0] * (self.max_seq_len - len(seq)) + list(seq)

    def fit(self, user_ids, item_ids, ratings=None, timestamps=None, **kwargs) -> "_SequentialBase":
        try:
            (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
                user_ids, item_ids, ratings if ratings is not None else np.ones(len(user_ids)),
                supported_modes=("triplet",))
        except Exception:
            pass
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        u = np.asarray(user_ids); it = np.asarray(item_ids)
        items = sorted(set(it.tolist()))
        self.item_map = {x: k + 1 for k, x in enumerate(items)}   # reserve 0 = pad
        self.reverse_item_map = {k + 1: x for k, x in enumerate(items)}
        self.iid_map = self.item_map
        self.num_items = len(items)
        self.uid_map = {}

        order = np.arange(len(u))
        if timestamps is not None:
            order = np.argsort(np.asarray(timestamps), kind="stable")
        seqs = {}
        for j in order:
            seqs.setdefault(u[j], []).append(self.item_map[it[j]])
        self.user_seq = seqs
        self.num_users = len(seqs)

        # sliding-window next-item training samples
        X, Y = [], []
        for s in seqs.values():
            for t in range(1, len(s)):
                X.append(self._pad(s[:t])); Y.append(s[t])
        if not X:
            raise ValueError("no sequential training samples (need >=2 interactions per user)")
        X = np.asarray(X, dtype=np.int64); Y = np.asarray(Y, dtype=np.int64)

        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = _SeqNet(self.num_items, self.embedding_dim, self._make_scorer()).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        bce = nn.BCEWithLogitsLoss()
        Xt = torch.as_tensor(X, device=dev); Yt = torch.as_tensor(Y, device=dev)
        n = Xt.shape[0]; rng = np.random.RandomState(self.seed)
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0; nb = 0
            for st in range(0, n, self.batch_size):
                idx = perm[st:st + self.batch_size]
                if idx.numel() < 2:
                    continue
                bx = Xt[idx]; pos = Yt[idx]
                neg = torch.randint(1, self.num_items + 1,
                                    (idx.numel(), self.num_negatives), device=dev)
                cand = torch.cat([pos.unsqueeze(1), neg], dim=1)          # [B, 1+k]
                logits = self.model.forward_scores(bx, cand)             # [B, 1+k]
                labels = torch.zeros_like(logits); labels[:, 0] = 1.0
                loss = bce(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if self.verbose and (ep + 1) % 5 == 0:
                logger.info(f"{self.MODEL} epoch {ep+1}/{self.epochs} loss={tot/max(1,nb):.4f}")

        self._dev = dev
        self.is_fitted = True
        self.model.eval()
        with torch.no_grad():
            probe = self._score_all_items(next(iter(seqs)))
            if float(np.std(probe)) < 1e-5:
                logger.warning("%s output collapsed (std~0).", self.MODEL)
        return self

    def _score_all_items(self, user_id) -> np.ndarray:
        seq = self.user_seq.get(user_id, [])
        bx = torch.as_tensor([self._pad(seq)], device=self._dev)
        cand = torch.arange(1, self.num_items + 1, device=self._dev).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            return self.model.forward_scores(bx, cand).squeeze(0).detach().cpu().numpy()

    def predict(self, user_id, item_id, **kwargs) -> float:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if item_id not in self.item_map:
            return 0.0
        seq = self.user_seq.get(user_id, [])
        bx = torch.as_tensor([self._pad(seq)], device=self._dev)
        cand = torch.as_tensor([[self.item_map[item_id]]], device=self._dev)
        self.model.eval()
        with torch.no_grad():
            return float(self.model.forward_scores(bx, cand).item())

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, **kwargs) -> List[Any]:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_seq:
            return []
        exclude = set(exclude_items or [])
        exclude |= {self.reverse_item_map[i] for i in self.user_seq[user_id]}  # seen
        scores = self._score_all_items(user_id)
        out = []
        for idx in np.argsort(-scores):
            iid = self.reverse_item_map[int(idx) + 1]
            if iid in exclude:
                continue
            out.append(iid)
            if len(out) >= top_k:
                break
        return out

    def save(self, path: Union[str, Path], **kwargs) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cfg": {"name": self.name, "embedding_dim": self.embedding_dim,
                    "max_seq_len": self.max_seq_len, "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size, "epochs": self.epochs,
                    "num_negatives": self.num_negatives, "reg": self.reg,
                    "device": self.device, "seed": self.seed},
            "item_map": self.item_map, "user_seq": self.user_seq,
            "num_items": self.num_items, "num_users": self.num_users,
            "state_dict": self.model.state_dict() if self.model else None,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "_SequentialBase":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        inst = cls(**ckpt["cfg"])
        inst.item_map = ckpt["item_map"]
        inst.reverse_item_map = {v: k for k, v in inst.item_map.items()}
        inst.iid_map = inst.item_map
        inst.user_seq = ckpt["user_seq"]; inst.num_items = ckpt["num_items"]
        inst.num_users = ckpt["num_users"]
        inst.model = _SeqNet(inst.num_items, inst.embedding_dim, inst._make_scorer())
        if ckpt["state_dict"] is not None:
            inst.model.load_state_dict(ckpt["state_dict"])
        inst.model.eval(); inst._dev = torch.device("cpu"); inst.device = "cpu"
        inst.is_fitted = True
        return inst


class GRU4Rec(_SequentialBase): MODEL = "GRU4Rec"
class Caser(_SequentialBase):   MODEL = "Caser"
class BST(_SequentialBase):     MODEL = "BST"
class DIN(_SequentialBase):     MODEL = "DIN"
class DIEN(_SequentialBase):    MODEL = "DIEN"
class NARM(_SequentialBase):    MODEL = "NARM"

__all__ = ["GRU4Rec", "Caser", "BST", "DIN", "DIEN", "NARM"]
