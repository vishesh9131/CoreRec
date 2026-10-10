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
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from corerec.device import resolve_device
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import InvalidDataError
from corerec.api.interactions import to_interactions

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
                 device: str = "auto",
                 seed: int = 42, trainable: bool = True, binarize: bool = True):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        self.hidden_dim = hidden_dim; self.latent_dim = latent_dim
        self.dropout = dropout; self.learning_rate = learning_rate
        self.batch_size = batch_size; self.epochs = epochs; self.beta = beta
        self.reg = reg; self.device = str(resolve_device(device)); self.seed = seed
        # Mult-VAE/DAE are defined on a 0/1 click matrix (Liang et al., 2018);
        # binarize=False keeps repeat counts, which is what fit() did before
        self.binarize = binarize
        self.model = None; self.user_map = {}; self.item_map = {}

    def fit(self, user_ids, item_ids, ratings=None, **kwargs) -> "_VAEBase":
        # one adapter for every input form (#78); it also rejects NaN/inf ratings
        events = to_interactions(user_ids, item_ids, ratings)
        if not len(events):
            raise InvalidDataError("Interactions must not be empty")
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        users_index, uidx = events.users, events.user_codes
        items_index, iidx = events.items, events.item_codes
        self.user_map = users_index.as_dict()
        self.item_map = items_index.as_dict()
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        self.num_users = len(users_index); self.num_items = len(items_index)
        self.R = csr_matrix((np.ones(len(uidx), np.float32), (uidx, iidx)),
                            shape=(self.num_users, self.num_items))
        if self.binarize:
            self.R.data[:] = 1.0  # csr_matrix summed the duplicate (user, item) events

        dev = resolve_device(self.device)
        self.model = _VAENet(self.num_items, self.hidden_dim, self.latent_dim,
                             self.dropout, self.VARIATIONAL).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        # Densify one batch at a time. The whole R on the device was
        # n_users * n_items floats: 8 GB at 100k x 20k.
        n = self.num_users
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n).numpy(); tot = 0.0; nb = 0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                x = torch.as_tensor(self.R[idx].toarray(), device=dev)
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
            if float(np.std(self._score_all_items(users_index.ids[0]))) < 1e-5:
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

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, *,
                  exclude_seen: bool = True, return_scores: bool = False,
                  **kwargs) -> Union[List[Any], List[Tuple[Any, float]]]:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or user_id not in self.user_map:
            return []
        exclude = set(exclude_items or [])
        scores = self._score_all_items(user_id).copy()
        if exclude_seen:
            scores[self.R[self.user_map[user_id]].indices] = -np.inf
        out = []
        for idx in np.argsort(-scores):
            if not np.isfinite(scores[idx]):
                continue
            iid = self.reverse_item_map[int(idx)]
            if iid in exclude:
                continue
            out.append((iid, float(scores[idx])) if return_scores else iid)
            if len(out) >= top_k:
                break
        return out

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Write a corerec_safe_v1 bundle: JSON config and ids, npz arrays, numeric tensor weights. The old torch.save checkpoint ran code on
        load (#75)."""
        from corerec.api.model_bundle import save_bundle

        # before any file is touched: an unfitted save used to truncate the old artifact (#101)
        self._check_fitted()
        R = self.R.tocsr()
        save_bundle(
            path, model_class=f"{type(self).__module__}.{type(self).__name__}",
            config={"name": self.name, "hidden_dim": self.hidden_dim,
                    "latent_dim": self.latent_dim, "dropout": self.dropout,
                    "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                    "epochs": self.epochs, "beta": self.beta, "reg": self.reg,
                    "device": self.device, "seed": self.seed, "binarize": self.binarize},
            state={"users": list(self.user_map), "items": list(self.item_map)},
            state_dict=self.model.state_dict() if self.model else None,
            arrays={"R_data": R.data, "R_indices": R.indices, "R_indptr": R.indptr,
                    "R_shape": np.asarray(R.shape)})

    @classmethod
    def load(cls, path: Union[str, Path], *, allow_pickle: bool = False, **kwargs) -> "_VAEBase":
        from corerec.api.model_bundle import is_safe_bundle, load_bundle

        if not is_safe_bundle(path):
            return cls._load_legacy(path, allow_pickle=allow_pickle)
        b = load_bundle(path, map_location="cpu", allow_pickle=allow_pickle)
        a = b["arrays"]
        inst = cls(**b["config"])
        inst._restore(
            {x: k for k, x in enumerate(b["state"]["users"])},
            {x: k for k, x in enumerate(b["state"]["items"])},
            csr_matrix((a["R_data"], a["R_indices"], a["R_indptr"]), shape=tuple(a["R_shape"])),
            b["state_dict"])
        return inst

    @classmethod
    def _load_legacy(cls, path, *, allow_pickle=False):
        from corerec.api.model_bundle import require_legacy_pickle

        require_legacy_pickle(path, allow_pickle)
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        # bundles from before binarize existed were trained on counts
        inst = cls(**{"binarize": False, **ckpt["cfg"]})
        inst._restore(ckpt["user_map"], ckpt["item_map"], ckpt["R"], ckpt["state_dict"])
        return inst

    def _restore(self, user_map, item_map, R, state_dict):
        self.user_map = user_map; self.item_map = item_map
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        self.num_users = len(user_map); self.num_items = len(item_map); self.R = R
        self.model = _VAENet(self.num_items, self.hidden_dim, self.latent_dim,
                             self.dropout, self.VARIATIONAL)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval(); self._dev = torch.device("cpu"); self.device = "cpu"; self.is_fitted = True


class MultVAE(_VAEBase):
    MODEL = "MultVAE"; VARIATIONAL = True


class MultiDAE(_VAEBase):
    MODEL = "MultiDAE"; VARIATIONAL = False


__all__ = ["MultVAE", "MultiDAE"]
