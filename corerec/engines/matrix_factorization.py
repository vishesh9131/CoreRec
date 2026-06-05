"""Production-grade embedding collaborative filtering: native ALS/WMF and item2vec.

A shared base (:class:`_EmbeddingCFBase`) holds user/item factor matrices and
provides the unified contract (predict/recommend/save/load via U @ V^T). Each model
implements ``_train_embeddings`` -> (U, V).

Models:
- ALS / WMF: implicit-feedback weighted matrix factorization (Hu 2008), solved by
  alternating least squares -- a strong, fast classical baseline.
- Item2Vec: skip-gram with negative sampling over co-interacted items (Barkan 2016);
  a user is represented by the mean of their item vectors.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from scipy.sparse import csr_matrix

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class _EmbeddingCFBase(BaseRecommender):
    MODEL = "ALS"

    def __init__(self, name: str = None, factors: int = 64, reg: float = 0.01,
                 iterations: int = 15, verbose: bool = False, seed: int = 42,
                 trainable: bool = True, **kwargs):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        self.factors = factors; self.reg = reg; self.iterations = iterations
        self.seed = seed
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.user_map = {}; self.item_map = {}

    def fit(self, user_ids, item_ids, ratings=None, **kwargs) -> "_EmbeddingCFBase":
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings if ratings is not None else np.ones(len(user_ids)),
            supported_modes=("triplet",))
        u = np.asarray(user_ids); it = np.asarray(item_ids); r = np.asarray(ratings, float)
        users = sorted(set(u.tolist())); items = sorted(set(it.tolist()))
        self.user_map = {x: k for k, x in enumerate(users)}
        self.item_map = {x: k for k, x in enumerate(items)}
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        self.num_users = len(users); self.num_items = len(items)
        uidx = np.fromiter((self.user_map[x] for x in u.tolist()), dtype=np.int64)
        iidx = np.fromiter((self.item_map[x] for x in it.tolist()), dtype=np.int64)
        self.R = csr_matrix((r, (uidx, iidx)), shape=(self.num_users, self.num_items))
        self.U, self.V = self._train_embeddings(uidx, iidx, r)
        if float(np.std(self.U[0] @ self.V.T)) < 1e-6:
            logger.warning("%s output collapsed (std~0).", self.MODEL)
        self.is_fitted = True
        return self

    def _train_embeddings(self, uidx, iidx, r):
        raise NotImplementedError

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
        with open(p, "wb") as f:
            pickle.dump({"U": self.U, "V": self.V, "R": self.R,
                         "user_map": self.user_map, "item_map": self.item_map,
                         "params": {"name": self.name, "factors": self.factors,
                                    "reg": self.reg, "iterations": self.iterations}}, f)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "_EmbeddingCFBase":
        with open(Path(path), "rb") as f:
            d = pickle.load(f)
        inst = cls(**d["params"])
        inst.U = d["U"]; inst.V = d["V"]; inst.R = d["R"]
        inst.user_map = d["user_map"]; inst.item_map = d["item_map"]
        inst.uid_map = inst.user_map; inst.iid_map = inst.item_map
        inst.reverse_item_map = {k: x for x, k in inst.item_map.items()}
        inst.num_users = len(inst.user_map); inst.num_items = len(inst.item_map)
        inst.is_fitted = True
        return inst


class ALS(_EmbeddingCFBase):
    """Implicit-feedback weighted MF via alternating least squares (Hu 2008)."""
    MODEL = "ALS"

    def __init__(self, name: str = None, factors: int = 64, reg: float = 10.0,
                 iterations: int = 20, alpha: float = 1.0, **kwargs):
        super().__init__(name=name, factors=factors, reg=reg, iterations=iterations,
                         alpha=alpha, **kwargs)

    def _train_embeddings(self, uidx, iidx, r):
        rng = np.random.RandomState(self.seed)
        f = self.factors
        X = rng.normal(0, 0.01, (self.num_users, f))
        Y = rng.normal(0, 0.01, (self.num_items, f))
        C = self.R.copy(); C.data = self.alpha * C.data          # Cui - 1 = alpha * r
        Ccsc = C.tocsc()
        I = self.reg * np.eye(f)

        def _solve(P, fixed, other_n):
            # P: sparse [A, B] confidence-1 (Cui-1); fixed: [B, f]
            G = fixed.T @ fixed                                   # Y^T Y
            out = np.zeros((P.shape[0], f))
            Pcsr = P.tocsr()
            for a in range(P.shape[0]):
                row = Pcsr[a]
                idx = row.indices
                if len(idx) == 0:
                    continue
                cu = row.data                                    # Cui - 1
                Yi = fixed[idx]                                  # [k, f]
                A = G + (Yi.T * cu) @ Yi + I                     # Y^T C Y + reg
                b = (Yi * (cu + 1.0)[:, None]).sum(0)            # Y^T C p  (p=1)
                out[a] = np.linalg.solve(A, b)
            return out

        for _ in range(self.iterations):
            X = _solve(C, Y, self.num_items)
            Y = _solve(Ccsc.T.tocsr(), X, self.num_users)
        return X, Y


class Item2Vec(_EmbeddingCFBase):
    """Skip-gram with negative sampling over co-interacted items (Barkan 2016)."""
    MODEL = "Item2Vec"

    def __init__(self, name: str = None, factors: int = 64, iterations: int = 20,
                 num_negatives: int = 5, learning_rate: float = 0.05, **kwargs):
        super().__init__(name=name, factors=factors, iterations=iterations,
                         num_negatives=num_negatives, learning_rate=learning_rate, **kwargs)

    def _train_embeddings(self, uidx, iidx, r):
        import torch
        torch.manual_seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # positive (center, context) pairs from each user's item set
        baskets = {}
        for u, i in zip(uidx.tolist(), iidx.tolist()):
            baskets.setdefault(u, []).append(i)
        centers, contexts = [], []
        for items in baskets.values():
            if len(items) < 2:
                continue
            arr = np.array(items)
            for c in arr:
                ctx = arr[arr != c]
                centers.extend([c] * len(ctx)); contexts.extend(ctx.tolist())
        if not centers:
            rng = np.random.RandomState(self.seed)
            return (rng.normal(0, .01, (self.num_users, self.factors)),
                    rng.normal(0, .01, (self.num_items, self.factors)))
        ce = torch.tensor(centers, device=dev); co = torch.tensor(contexts, device=dev)
        Win = (torch.randn(self.num_items, self.factors, device=dev) * 0.01).requires_grad_(True)
        Wout = (torch.randn(self.num_items, self.factors, device=dev) * 0.01).requires_grad_(True)
        opt = torch.optim.Adam([Win, Wout], lr=self.learning_rate)
        n = ce.shape[0]; bs = 16384
        for _ in range(self.iterations):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                c = Win[ce[idx]]; pos = Wout[co[idx]]
                neg = Wout[torch.randint(0, self.num_items, (idx.numel(), self.num_negatives), device=dev)]
                pl = torch.nn.functional.logsigmoid((c * pos).sum(1))
                nl = torch.nn.functional.logsigmoid(-(c.unsqueeze(1) * neg).sum(2)).sum(1)
                loss = -(pl + nl).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        V = Win.detach().cpu().numpy()
        # skip-gram vectors rank by cosine, so L2-normalize the item table
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        # user vector = mean of interacted (normalized) item vectors
        U = np.zeros((self.num_users, self.factors), dtype=np.float32)
        Rcsr = self.R.tocsr()
        for u in range(self.num_users):
            idx = Rcsr[u].indices
            if len(idx):
                U[u] = V[idx].mean(0)
        return U, V


__all__ = ["ALS", "Item2Vec"]
