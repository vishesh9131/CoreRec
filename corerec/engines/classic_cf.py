"""Production-grade classic collaborative-filtering models.

A shared base (:class:`_ClassicCFBase`) builds the sparse user-item matrix and
provides the unified contract -- ``fit``/``predict``/vectorized ``recommend``/
``save``/``load``. Each model defines how items are scored from a user's history.

Models: ItemKNN (item-item cosine kNN), UserKNN (user-user cosine kNN), and EASE
(embarrassingly shallow auto-encoder; Steck 2019 -- a closed-form item-item model
that is frequently competitive with deep methods).
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


class _ClassicCFBase(BaseRecommender):
    MODEL = "ItemKNN"

    def __init__(self, name: str = None, top_k_neighbors: int = 100,
                 reg: float = 250.0, shrink: float = 0.0, verbose: bool = False,
                 trainable: bool = True):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        self.top_k_neighbors = top_k_neighbors
        self.reg = reg          # EASE L2
        self.shrink = shrink    # kNN shrinkage
        self.user_map = {}
        self.item_map = {}

    # -- contract: fit -------------------------------------------------- #
    def fit(self, user_ids, item_ids, ratings=None, **kwargs) -> "_ClassicCFBase":
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings if ratings is not None else np.ones(len(user_ids)),
            supported_modes=("triplet",))
        u = np.asarray(user_ids); it = np.asarray(item_ids)
        r = np.asarray(ratings, dtype=float)
        users = sorted(set(u.tolist())); items = sorted(set(it.tolist()))
        self.user_map = {x: k for k, x in enumerate(users)}
        self.item_map = {x: k for k, x in enumerate(items)}
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        self.num_users = len(users); self.num_items = len(items)
        uidx = np.fromiter((self.user_map[x] for x in u.tolist()), dtype=np.int64)
        iidx = np.fromiter((self.item_map[x] for x in it.tolist()), dtype=np.int64)
        self.R = csr_matrix((r, (uidx, iidx)), shape=(self.num_users, self.num_items))
        self._fit_model()
        self.is_fitted = True
        return self

    def _fit_model(self):
        raise NotImplementedError

    # -- contract: scoring --------------------------------------------- #
    def _score_all_items(self, user_id) -> np.ndarray:
        raise NotImplementedError

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
        uidx = self.user_map[user_id]
        scores = self._score_all_items(user_id).copy()
        scores[self.R[uidx].indices] = -np.inf          # exclude already seen
        out = []
        for idx in np.argsort(-scores):
            iid = self.reverse_item_map[int(idx)]
            if iid in exclude:
                continue
            out.append(iid)
            if len(out) >= top_k:
                break
        return out

    # -- contract: persistence ----------------------------------------- #
    def save(self, path: Union[str, Path], **kwargs) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"cls": self.__class__.__name__, "user_map": self.user_map,
                         "item_map": self.item_map, "R": self.R,
                         "state": self._state(),
                         "params": {"top_k_neighbors": self.top_k_neighbors,
                                    "reg": self.reg, "shrink": self.shrink,
                                    "name": self.name}}, f)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "_ClassicCFBase":
        with open(Path(path), "rb") as f:
            d = pickle.load(f)
        inst = cls(**d["params"])
        inst.user_map = d["user_map"]; inst.item_map = d["item_map"]
        inst.uid_map = inst.user_map; inst.iid_map = inst.item_map
        inst.reverse_item_map = {k: x for x, k in inst.item_map.items()}
        inst.num_users = len(inst.user_map); inst.num_items = len(inst.item_map)
        inst.R = d["R"]; inst._set_state(d["state"]); inst.is_fitted = True
        return inst

    def _state(self):
        return {}

    def _set_state(self, state):
        pass


def _cosine_gram(R, shrink):
    G = (R.T @ R).toarray().astype(np.float64)          # [I, I] co-occurrence
    norms = np.sqrt(np.maximum(np.diag(G), 1e-12))
    S = G / (np.outer(norms, norms) + shrink + 1e-12)
    np.fill_diagonal(S, 0.0)
    return S


def _topk_rows(S, k):
    if k and k < S.shape[1]:
        for r in range(S.shape[0]):
            row = S[r]
            cut = np.argpartition(row, -k)[:-k]
            row[cut] = 0.0
    return S


class ItemKNN(_ClassicCFBase):
    MODEL = "ItemKNN"

    def _fit_model(self):
        S = _cosine_gram(self.R, self.shrink)
        self.S = _topk_rows(S, self.top_k_neighbors).astype(np.float32)

    def _score_all_items(self, user_id):
        return np.asarray(self.R[self.user_map[user_id]] @ self.S).ravel()

    def _state(self): return {"S": self.S}
    def _set_state(self, s): self.S = s["S"]


class UserKNN(_ClassicCFBase):
    MODEL = "UserKNN"

    def _fit_model(self):
        G = (self.R @ self.R.T).toarray().astype(np.float64)   # user-user
        norms = np.sqrt(np.maximum(np.diag(G), 1e-12))
        Su = G / (np.outer(norms, norms) + self.shrink + 1e-12)
        np.fill_diagonal(Su, 0.0)
        self.Su = _topk_rows(Su, self.top_k_neighbors).astype(np.float32)

    def _score_all_items(self, user_id):
        u = self.user_map[user_id]
        return np.asarray(self.Su[u] @ self.R).ravel()

    def _state(self): return {"Su": self.Su}
    def _set_state(self, s): self.Su = s["Su"]


class EASE(_ClassicCFBase):
    """Closed-form item-item model: B = -P/diag(P), P=(R^T R + reg I)^-1."""
    MODEL = "EASE"

    def _fit_model(self):
        G = (self.R.T @ self.R).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)

    def _score_all_items(self, user_id):
        return np.asarray(self.R[self.user_map[user_id]] @ self.B).ravel()

    def _state(self): return {"B": self.B}
    def _set_state(self, s): self.B = s["B"]


class SLIM(_ClassicCFBase):
    """Sparse Linear Methods (Ning & Karypis 2011): a sparse, non-negative
    item-item weight matrix learned by elastic-net regression per item."""
    MODEL = "SLIM"

    def __init__(self, name: str = None, l1_ratio: float = 0.1, alpha: float = 0.1,
                 max_iter: int = 50, **kwargs):
        super().__init__(name=name, **kwargs)
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.max_iter = max_iter

    def _fit_model(self):
        from sklearn.linear_model import ElasticNet
        R = self.R.tocsc()
        n = self.num_items
        W = np.zeros((n, n), dtype=np.float32)
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, positive=True,
                           fit_intercept=False, copy_X=False, max_iter=self.max_iter,
                           tol=1e-3)
        for j in range(n):
            target = R[:, j].toarray().ravel()
            col_backup = R[:, j].copy()
            R[:, j] = 0.0                                   # exclude self
            model.fit(R, target)
            W[:, j] = model.coef_
            R[:, j] = col_backup
        np.fill_diagonal(W, 0.0)
        self.W = W

    def _score_all_items(self, user_id):
        return np.asarray(self.R[self.user_map[user_id]] @ self.W).ravel()

    def _state(self): return {"W": self.W}
    def _set_state(self, s): self.W = s["W"]


__all__ = ["ItemKNN", "UserKNN", "EASE", "SLIM"]
