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
import scipy.sparse as sp
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
        # float32 to match the item-item matrices: a float64 user row against a
        # float32 [I, I] matrix made numpy upcast-copy the whole matrix per call
        self.R = csr_matrix((r, (uidx, iidx)), shape=(self.num_users, self.num_items),
                            dtype=np.float32)
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
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or user_id not in self.user_map:
            return []
        exclude = set(exclude_items or [])
        uidx = self.user_map[user_id]
        scores = self._score_all_items(user_id).copy()
        scores[self.R[uidx].indices] = -np.inf          # exclude already seen
        out = []
        for idx in np.argsort(-scores):
            if not np.isfinite(scores[idx]):
                continue
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


def _row_times(R, row, M) -> np.ndarray:
    """R[row] @ M touching only the rows of M the user interacted with."""
    lo, hi = R.indptr[row], R.indptr[row + 1]
    idx, vals = R.indices[lo:hi], R.data[lo:hi].astype(np.float32)
    if sp.issparse(M):
        return np.asarray((csr_matrix((vals, idx, [0, len(idx)]), shape=(1, M.shape[0])) @ M)
                          .todense()).ravel()
    return vals @ M[idx]  # dense, e.g. EASE; also old pickles with a dense S


def _sparse_cosine_topk(X, shrink, k, max_block_entries=50_000_000):
    """Top-k cosine neighbours of each column of X, as a sparse [n, n] CSR.

    Built a block of rows at a time and pruned to k before the next block, so
    peak memory is ~max_block_entries floats however large n gets. The dense
    version this replaced allocated n*n (40 GB at 100k items); a one-shot sparse
    X^T X isn't enough either, since popular items co-occur with nearly everything.
    """
    X = X.tocsc()
    n = X.shape[1]
    Xt = X.T.tocsr()
    norms = np.sqrt(np.maximum(np.asarray(X.multiply(X).sum(axis=0)).ravel(), 1e-12))
    block = max(1, max_block_entries // max(n, 1))
    rows, cols, data = [], [], []
    for b0 in range(0, n, block):
        G = (Xt[b0:b0 + block] @ X).tocsr()
        # whole block at once; a per-row python loop here dominated fit time at 100k
        r = b0 + np.repeat(np.arange(G.shape[0]), np.diff(G.indptr))
        c = G.indices
        d = G.data / (norms[r] * norms[c] + shrink + 1e-12)
        keep = c != r
        r, c, d = r[keep], c[keep], d[keep]
        if k:
            # only rows with more than k neighbours need pruning; sort just those,
            # by row then score high->low, using one float key (|d| < 1, so row*4
            # keeps rows apart) and keep each row's first k
            counts = np.bincount(r - b0, minlength=G.shape[0])
            over = np.flatnonzero((counts > k)[r - b0])
            if len(over):
                order = over[np.argsort((r[over] - b0) * 4.0 - d[over], kind="stable")]
                n_over = counts[counts > k]
                rank = np.arange(len(order)) - np.repeat(np.cumsum(n_over) - n_over, n_over)
                keep = np.ones(len(r), bool)
                keep[over] = False
                keep[order[rank < k]] = True
                r, c, d = r[keep], c[keep], d[keep]
        rows.append(r); cols.append(c); data.append(d)
    if not rows:
        return csr_matrix((n, n), dtype=np.float32)
    return csr_matrix((np.concatenate(data).astype(np.float32),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


class ItemKNN(_ClassicCFBase):
    MODEL = "ItemKNN"

    def _fit_model(self):
        self.S = _sparse_cosine_topk(self.R, self.shrink, self.top_k_neighbors)

    def _score_all_items(self, user_id):
        return _row_times(self.R, self.user_map[user_id], self.S)

    def _state(self): return {"S": self.S}
    def _set_state(self, s): self.S = s["S"]


class UserKNN(_ClassicCFBase):
    MODEL = "UserKNN"

    def _fit_model(self):
        # user-user neighbours: cosine over the columns of R^T
        self.Su = _sparse_cosine_topk(self.R.T.tocsr(), self.shrink, self.top_k_neighbors)

    def _score_all_items(self, user_id):
        return _row_times(self.Su, self.user_map[user_id], self.R)

    def _state(self): return {"Su": self.Su}
    def _set_state(self, s): self.Su = csr_matrix(s["Su"])  # old pickles hold a dense Su


class EASE(_ClassicCFBase):
    """Closed-form item-item model: B = -P/diag(P), P=(R^T R + reg I)^-1."""
    MODEL = "EASE"

    def _fit_model(self):
        # EASE is a dense [I, I] inverse by construction; say so up front
        # rather than dying in the allocator
        need_gb = 3 * 8 * self.num_items ** 2 / 1e9
        if need_gb > 64:
            raise MemoryError(
                f"EASE on {self.num_items:,} items needs ~{need_gb:.0f} GB (dense item x item "
                "inverse). Use ItemKNN or ALS at this catalogue size, or prune rare items.")
        G = (self.R.T @ self.R).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)

    def _score_all_items(self, user_id):
        return _row_times(self.R, self.user_map[user_id], self.B)

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
        R = self.R.tocsc().astype(np.float64)
        n = self.num_items
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, positive=True,
                           fit_intercept=False, copy_X=False, max_iter=self.max_iter,
                           tol=1e-3)
        rows, cols, vals = [], [], []
        for j in range(n):
            lo, hi = R.indptr[j], R.indptr[j + 1]
            target = np.zeros(R.shape[0])
            target[R.indices[lo:hi]] = R.data[lo:hi]
            # Zero column j in place to exclude self. R[:, j] = 0 rebuilt the
            # sparse structure on every item -- SLIM couldn't finish 4k items in 10 min.
            saved = R.data[lo:hi].copy()
            R.data[lo:hi] = 0.0
            model.fit(R, target)
            R.data[lo:hi] = saved
            nz = np.flatnonzero(model.coef_)
            rows.append(nz); cols.append(np.full(len(nz), j)); vals.append(model.coef_[nz])
        W = csr_matrix((np.concatenate(vals).astype(np.float32),
                        (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        W.setdiag(0.0); W.eliminate_zeros()
        self.W = W

    def _score_all_items(self, user_id):
        return _row_times(self.R, self.user_map[user_id], self.W)

    def _state(self): return {"W": self.W}
    def _set_state(self, s): self.W = s["W"]


__all__ = ["ItemKNN", "UserKNN", "EASE", "SLIM"]
