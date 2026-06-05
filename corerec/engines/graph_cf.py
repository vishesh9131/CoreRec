"""Production-grade graph collaborative filtering: NGCF (Wang 2019).

Sparse normalized-adjacency propagation with per-layer transformation matrices and
non-linearity, concatenated layer embeddings, BPR training. Conforms to the unified
contract (fit/predict/recommend/save/load) with a sparse operator that scales
(no dense Laplacian). Complements the production LightGCN.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


def _norm_adj_selfloop(users, items, n_users, n_items, device):
    u = torch.as_tensor(users, dtype=torch.long)
    i = torch.as_tensor(items, dtype=torch.long) + n_users
    N = n_users + n_items
    rows = torch.cat([u, i, torch.arange(N)])
    cols = torch.cat([i, u, torch.arange(N)])     # + self loops
    deg = torch.zeros(N); deg.index_add_(0, rows, torch.ones(rows.shape[0]))
    dinv = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = dinv[rows] * dinv[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, N)).coalesce().to(device)


class _NGCFNet(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, dropout):
        super().__init__()
        self.E = nn.Parameter(torch.empty(n_users + n_items, dim))
        nn.init.normal_(self.E, std=0.1)
        self.W1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

    def propagate(self, A):
        E = self.E
        outs = [E]
        for l in range(self.n_layers):
            side = torch.sparse.mm(A, E)
            E = F.leaky_relu(self.W1[l](E + side) + self.W2[l](side * E))
            E = self.drop(E)
            outs.append(F.normalize(E, dim=1))
        return torch.cat(outs, dim=1)     # [N, dim*(L+1)]


class NGCF(BaseRecommender):
    MODEL = "NGCF"

    def __init__(self, name: str = "NGCF", embedding_dim: int = 64, n_layers: int = 3,
                 dropout: float = 0.1, learning_rate: float = 1e-3, reg: float = 1e-4,
                 batch_size: int = 8192, epochs: int = 200, verbose: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42, trainable: bool = True):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim; self.n_layers = n_layers
        self.dropout = dropout; self.learning_rate = learning_rate; self.reg = reg
        self.batch_size = batch_size; self.epochs = epochs
        self.device = device; self.seed = seed
        self.model = None; self.user_map = {}; self.item_map = {}

    def fit(self, user_ids, item_ids, ratings=None, **kwargs) -> "NGCF":
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

        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        A = _norm_adj_selfloop(uidx, iidx, self.num_users, self.num_items, dev)
        self.model = _NGCFNet(self.num_users, self.num_items, self.embedding_dim,
                              self.n_layers, self.dropout).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        ut = torch.as_tensor(uidx, device=dev); itt = torch.as_tensor(iidx, device=dev)
        n = ut.shape[0]
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0; nb = 0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                allE = self.model.propagate(A)
                U = allE[:self.num_users]; V = allE[self.num_users:]
                bu = ut[idx]; bi = itt[idx]
                bj = torch.randint(0, self.num_items, (bu.shape[0],), device=dev)
                pu = U[bu]; pi = V[bi]; pj = V[bj]
                x = (pu * pi).sum(1) - (pu * pj).sum(1)
                loss = -F.logsigmoid(x).mean()
                loss = loss + self.reg * (pu.pow(2).sum() + pi.pow(2).sum() + pj.pow(2).sum()) / bu.shape[0]
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if self.verbose and (ep + 1) % 20 == 0:
                logger.info(f"NGCF epoch {ep+1}/{self.epochs} loss={tot/max(1,nb):.4f}")
        self.model.eval()
        with torch.no_grad():
            allE = self.model.propagate(A)
            self.U = allE[:self.num_users].detach().cpu().numpy()
            self.V = allE[self.num_users:].detach().cpu().numpy()
        if float(np.std(self.U[0] @ self.V.T)) < 1e-5:
            logger.warning("NGCF output collapsed (std~0).")
        self.is_fitted = True
        return self

    def _score_all_items(self, user_id) -> np.ndarray:
        return self.U[self.user_map[user_id]] @ self.V.T

    def predict(self, user_id, item_id, **kwargs) -> float:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        return float(self.U[self.user_map[user_id]] @ self.V[self.item_map[item_id]])

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, **kwargs) -> List[Any]:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map:
            return []
        exclude = set(exclude_items or [])
        scores = self._score_all_items(user_id)
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
        np.savez(str(p) + ".npz", U=self.U, V=self.V)
        import json
        with open(str(p) + ".json", "w") as f:
            json.dump({"user_map": {str(k): v for k, v in self.user_map.items()},
                       "item_map": {str(k): v for k, v in self.item_map.items()},
                       "name": self.name, "embedding_dim": self.embedding_dim,
                       "n_layers": self.n_layers}, f)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "NGCF":
        import json
        with open(str(path) + ".json") as f:
            d = json.load(f)
        inst = cls(name=d["name"], embedding_dim=d["embedding_dim"], n_layers=d["n_layers"])
        arr = np.load(str(path) + ".npz")
        inst.U = arr["U"]; inst.V = arr["V"]
        def _key(k):
            try: return int(k)
            except ValueError: return k
        inst.user_map = {_key(k): v for k, v in d["user_map"].items()}
        inst.item_map = {_key(k): v for k, v in d["item_map"].items()}
        inst.uid_map = inst.user_map; inst.iid_map = inst.item_map
        inst.reverse_item_map = {v: k for k, v in inst.item_map.items()}
        inst.num_users = len(inst.user_map); inst.num_items = len(inst.item_map)
        inst.is_fitted = True
        return inst


__all__ = ["NGCF"]
