"""Production-grade variational/denoising auto-encoder collaborative filtering.

A shared base (:class:`_VAEBase`) builds the sparse user-item matrix and provides
the unified contract -- ``fit`` (per-user reconstruction with the multinomial
likelihood), ``predict``, vectorized ``recommend``, ``save``/``load``, device flag
and a collapse guard.

Models: MultVAE (Liang 2018, variational with KL annealing) and MultiDAE
(denoising auto-encoder). Both are strong, widely-cited CF auto-encoders.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class _Encoder(nn.Module):
    def __init__(self, n_items, hidden, latent, variational):
        super().__init__()
        self.variational = variational
        self.net = nn.Sequential(nn.Linear(n_items, hidden), nn.Tanh())
        self.out = nn.Linear(hidden, latent * 2 if variational else latent)
        self.latent = latent

    def forward(self, x):
        h = self.out(self.net(x))
        if self.variational:
            mu, logvar = h[:, :self.latent], h[:, self.latent:]
            return mu, logvar
        return h, None


class _VAENet(nn.Module):
    def __init__(self, n_items, hidden, latent, dropout, variational):
        super().__init__()
        self.enc = _Encoder(n_items, hidden, latent, variational)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, n_items))
        self.drop = nn.Dropout(dropout)
        self.variational = variational

    def forward(self, x, sample=True):
        x = F.normalize(x, dim=1)
        x = self.drop(x)
        mu, logvar = self.enc(x)
        if self.variational and sample and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.dec(z), mu, logvar


class _VAEBase(BaseRecommender):
    MODEL = "MultVAE"
    VARIATIONAL = True

    def __init__(self, name: str = None, hidden_dim: int = 600, latent_dim: int = 200,
                 dropout: float = 0.5, learning_rate: float = 1e-3, batch_size: int = 256,
                 epochs: int = 50, beta: float = 0.2, reg: float = 0.0,
                 verbose: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42, trainable: bool = True):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        self.hidden_dim = hidden_dim; self.latent_dim = latent_dim
        self.dropout = dropout; self.learning_rate = learning_rate
        self.batch_size = batch_size; self.epochs = epochs; self.beta = beta
        self.reg = reg; self.device = device; self.seed = seed
        self.model = None; self.user_map = {}; self.item_map = {}

    def fit(self, user_ids, item_ids, ratings=None, **kwargs) -> "_VAEBase":
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings if ratings is not None else np.ones(len(user_ids)),
            supported_modes=("triplet",))
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        u = np.asarray(user_ids); it = np.asarray(item_ids)
        users = sorted(set(u.tolist())); items = sorted(set(it.tolist()))
        self.user_map = {x: k for k, x in enumerate(users)}
        self.item_map = {x: k for k, x in enumerate(items)}
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        self.num_users = len(users); self.num_items = len(items)
        uidx = np.fromiter((self.user_map[x] for x in u.tolist()), dtype=np.int64)
        iidx = np.fromiter((self.item_map[x] for x in it.tolist()), dtype=np.int64)
        self.R = csr_matrix((np.ones(len(uidx), np.float32), (uidx, iidx)),
                            shape=(self.num_users, self.num_items))

        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = _VAENet(self.num_items, self.hidden_dim, self.latent_dim,
                             self.dropout, self.VARIATIONAL).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        Rd = torch.as_tensor(self.R.toarray(), device=dev)
        n = self.num_users
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0; nb = 0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                x = Rd[idx]
                logits, mu, logvar = self.model(x)
                ll = -(F.log_softmax(logits, 1) * x).sum(1).mean()
                if self.VARIATIONAL:
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
                    loss = ll + self.beta * kl
                else:
                    loss = ll
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if self.verbose and (ep + 1) % 10 == 0:
                logger.info(f"{self.MODEL} epoch {ep+1}/{self.epochs} loss={tot/max(1,nb):.4f}")
        self._dev = dev; self.is_fitted = True
        self.model.eval()
        with torch.no_grad():
            if float(np.std(self._score_all_items(users[0]))) < 1e-5:
                logger.warning("%s output collapsed (std~0).", self.MODEL)
        return self

    def _score_all_items(self, user_id) -> np.ndarray:
        x = torch.as_tensor(self.R[self.user_map[user_id]].toarray(), device=self._dev).float()
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(x, sample=False)
        return logits.squeeze(0).detach().cpu().numpy()

    def predict(self, user_id, item_id, **kwargs) -> float:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        return float(self._score_all_items(user_id)[self.item_map[item_id]])

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, **kwargs) -> List[Any]:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map:
            return []
        exclude = set(exclude_items or [])
        scores = self._score_all_items(user_id).copy()
        scores[self.R[self.user_map[user_id]].indices] = -np.inf
        out = []
        for idx in np.argsort(-scores):
            iid = self.reverse_item_map[int(idx)]
            if iid in exclude:
                continue
            out.append(iid)
            if len(out) >= top_k:
                break
        return out

    def save(self, path: Union[str, Path], **kwargs) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": {"name": self.name, "hidden_dim": self.hidden_dim,
                            "latent_dim": self.latent_dim, "dropout": self.dropout,
                            "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                            "epochs": self.epochs, "beta": self.beta, "reg": self.reg,
                            "device": self.device, "seed": self.seed},
                    "user_map": self.user_map, "item_map": self.item_map,
                    "num_users": self.num_users, "num_items": self.num_items,
                    "R": self.R, "state_dict": self.model.state_dict() if self.model else None}, p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "_VAEBase":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        inst = cls(**ckpt["cfg"])
        inst.user_map = ckpt["user_map"]; inst.item_map = ckpt["item_map"]
        inst.uid_map = inst.user_map; inst.iid_map = inst.item_map
        inst.reverse_item_map = {k: x for x, k in inst.item_map.items()}
        inst.num_users = ckpt["num_users"]; inst.num_items = ckpt["num_items"]; inst.R = ckpt["R"]
        inst.model = _VAENet(inst.num_items, inst.hidden_dim, inst.latent_dim,
                             inst.dropout, inst.VARIATIONAL)
        if ckpt["state_dict"] is not None:
            inst.model.load_state_dict(ckpt["state_dict"])
        inst.model.eval(); inst._dev = torch.device("cpu"); inst.device = "cpu"; inst.is_fitted = True
        return inst


class MultVAE(_VAEBase):
    MODEL = "MultVAE"; VARIATIONAL = True


class MultiDAE(_VAEBase):
    MODEL = "MultiDAE"; VARIATIONAL = False


__all__ = ["MultVAE", "MultiDAE"]
