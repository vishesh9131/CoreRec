"""
Turn any torch ``nn.Module`` into a CoreRec model.

    from corerec.nn import Recommender
    from corerec.nn.models import MatrixFactorization

    rec = Recommender(MatrixFactorization, {"dim": 64}, loss="bpr", epochs=20)
    rec.fit(df)                      # user_id, item_id[, rating, timestamp]
    rec.recommend("u42", top_k=10)

The result is an ordinary CoreRec recommender, so ModelServer, ModelLoader,
Evaluator and corerec.export.to_onnx work on it unchanged.

The module contract
-------------------
``forward(query, items) -> scores``

- ``query``: LongTensor [B] of user indices (``inputs="user"``), or
  [B, max_len] item-index histories, oldest first, left-padded with 0
  (``inputs="history"``).
- ``items``: LongTensor [B, K] of item indices.
- returns: FloatTensor [B, K], higher = better.

Item indices run 1..n_items; 0 is padding, so size item tables
``nn.Embedding(n_items + 1, d, padding_idx=0)``. User indices run 0..n_users-1.
The module is built as ``module_cls(n_users=..., n_items=..., max_len=...,
**module_kwargs)``, passing only the arguments its __init__ accepts.

Optionally define ``score_all(query) -> [B, n_items + 1]`` (column 0 ignored)
for fast full-catalogue scoring; otherwise ``forward`` is called on every item.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from corerec.api.base_recommender import BaseRecommender
from corerec.device import resolve_device
from corerec.nn.losses import LOSSES

logger = logging.getLogger(__name__)


class Recommender(BaseRecommender):
    """Train, serve and persist a user-defined torch module.

    Args:
        module_cls: the nn.Module class (not an instance), see the module docstring.
        module_kwargs: extra constructor arguments; keep them JSON-like so the
            model can be saved and rebuilt.
        inputs: "user" or "history" -- what the module's query is.
        loss: "bpr", "bce", "sampled_softmax" or a callable(scores[B, 1+K]) -> loss.
        num_negatives: uniformly sampled negatives per positive.
        max_len: history length for inputs="history".
    """

    def __init__(self, module_cls: type, module_kwargs: Optional[Dict[str, Any]] = None, *,
                 inputs: str = "user", loss: Union[str, Callable] = "bpr",
                 num_negatives: int = 1, epochs: int = 10, batch_size: int = 1024,
                 lr: float = 1e-3, weight_decay: float = 0.0, max_len: int = 50,
                 device: str = "auto", seed: int = 0, verbose: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name or module_cls.__name__, verbose=verbose)
        if inputs not in ("user", "history"):
            raise ValueError("inputs must be 'user' or 'history'")
        if isinstance(loss, str) and loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)} or a callable")
        self.module_cls, self.module_kwargs = module_cls, dict(module_kwargs or {})
        self.inputs, self.loss = inputs, loss
        self.num_negatives, self.epochs, self.batch_size = num_negatives, epochs, batch_size
        self.lr, self.weight_decay, self.max_len = lr, weight_decay, max_len
        self.device, self.seed = resolve_device(device), seed
        self.model: Optional[nn.Module] = None
        self.history_: List[float] = []
        self.val_history_: List[float] = []

    # -- data ----------------------------------------------------------- #
    def _index(self, user_ids, item_ids, ratings, timestamps):
        u = pd.Series(list(user_ids), dtype=object)
        i = pd.Series(list(item_ids), dtype=object)
        keep = np.ones(len(u), bool) if ratings is None else np.asarray(ratings, float) > 0
        if timestamps is not None:
            order = np.argsort(np.asarray(timestamps), kind="stable")
            u, i, keep = u.iloc[order], i.iloc[order], keep[order]
        u, i = u[keep].reset_index(drop=True), i[keep].reset_index(drop=True)
        ucodes, users = pd.factorize(u)
        icodes, items = pd.factorize(i)
        self._users, self._items = list(users), list(items)
        self.user_map = {x: k for k, x in enumerate(self._users)}
        self.item_map = {x: k + 1 for k, x in enumerate(self._items)}  # 0 = padding
        self.uid_map, self.iid_map = self.user_map, self.item_map
        self.num_users, self.num_items = len(users), len(items)
        icodes = icodes + 1
        self._seen = sp.csr_matrix((np.ones(len(ucodes), np.float32), (ucodes, icodes)),
                                   shape=(self.num_users, self.num_items + 1))
        self._seen.sum_duplicates()
        # per-user item sequence in event order (stable sort keeps time order)
        order = np.argsort(ucodes, kind="stable")
        bounds = np.searchsorted(ucodes[order], np.arange(self.num_users + 1))
        self._sequences = [icodes[order][bounds[k]:bounds[k + 1]] for k in range(self.num_users)]
        return ucodes, icodes

    def _histories(self, rows: List[np.ndarray]) -> np.ndarray:
        x = np.zeros((len(rows), self.max_len), dtype=np.int64)
        for r, seq in enumerate(rows):
            seq = seq[-self.max_len:]
            if len(seq):
                x[r, -len(seq):] = seq
        return x

    def _query(self, uidx: np.ndarray) -> torch.Tensor:
        if self.inputs == "user":
            q = np.asarray(uidx, dtype=np.int64)
        else:
            q = self._histories([self._sequences[k] for k in uidx])
        return torch.as_tensor(q, device=self.device)

    # -- model ---------------------------------------------------------- #
    def _build(self) -> nn.Module:
        offered = {"n_users": self.num_users, "n_items": self.num_items, "max_len": self.max_len}
        params = inspect.signature(self.module_cls.__init__).parameters
        takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in offered.items() if takes_all or k in params}
        return self.module_cls(**kwargs, **self.module_kwargs).to(self.device)

    def _score_all(self, query: torch.Tensor) -> torch.Tensor:
        """[B, n_items] scores for items 1..n_items."""
        if hasattr(self.model, "score_all"):
            return self.model.score_all(query)[:, 1:]
        all_items = torch.arange(1, self.num_items + 1, device=query.device)
        chunks = []
        for c in all_items.split(4096):
            # broadcast without arithmetic on the batch size, so it exports
            items = c.unsqueeze(0) + torch.zeros_like(query.reshape(query.shape[0], -1)[:, :1])
            chunks.append(self.model(query, items))
        return torch.cat(chunks, dim=1)

    # -- contract: fit ---------------------------------------------------- #
    def fit(self, user_ids, item_ids, ratings=None, timestamps=None, validation=None,
            patience: Optional[int] = None, **kwargs) -> "Recommender":
        """One row per interaction; rows with rating <= 0 are dropped.

        Events are taken in the order given, or sorted by ``timestamps``.

        validation: held-out interactions (DataFrame with user_id/item_id, or
            (user, item) pairs). NDCG@10 on it goes in ``val_history_`` after
            each epoch, and the best epoch's weights are kept.
        patience: stop after this many epochs without a better NDCG@10.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ucodes, icodes = self._index(user_ids, item_ids, ratings, timestamps)
        self.model = self._build()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = LOSSES[self.loss] if isinstance(self.loss, str) else self.loss

        if self.inputs == "user":
            ex_user, ex_target, ex_pos = ucodes, icodes, None
        else:
            # every event after a user's first, predicted from what came before it
            lens = np.array([len(s) for s in self._sequences])
            ex_user = np.repeat(np.arange(self.num_users), np.maximum(lens - 1, 0))
            ex_pos = np.concatenate([np.arange(1, n) for n in lens if n > 1] or [np.zeros(0, int)])
            ex_target = np.array([self._sequences[u][t] for u, t in zip(ex_user, ex_pos)], np.int64)
        n = len(ex_target)
        if n == 0:
            raise ValueError("no training examples: need at least two events per user for "
                             "inputs='history', or one interaction for inputs='user'")

        self.history_, self.val_history_ = [], []
        best_state, stale = None, 0
        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(n)
            total = 0.0
            for b0 in range(0, n, self.batch_size):
                idx = perm[b0:b0 + self.batch_size]
                if self.inputs == "user":
                    query = torch.as_tensor(ex_user[idx], device=self.device)
                else:
                    query = torch.as_tensor(self._histories(
                        [self._sequences[u][:t] for u, t in zip(ex_user[idx], ex_pos[idx])]),
                        device=self.device)
                neg = rng.integers(1, self.num_items + 1, (len(idx), self.num_negatives))
                items = torch.as_tensor(np.concatenate([ex_target[idx, None], neg], 1),
                                        device=self.device)
                scores = self.model(query, items)
                if scores.shape != items.shape:
                    raise ValueError(f"{self.module_cls.__name__}.forward returned shape "
                                     f"{tuple(scores.shape)}, expected {tuple(items.shape)}")
                loss = loss_fn(scores)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            self.history_.append(total / n)
            if self.verbose:
                logger.info("%s epoch %d/%d loss %.4f", self.name, epoch + 1, self.epochs,
                            self.history_[-1])
            if validation is not None:
                from corerec.evaluation.evaluate import evaluate

                self.model.eval()
                self.is_fitted = True  # evaluate() goes through recommend()
                ndcg = evaluate(self, validation, k=10)["NDCG@10"]
                self.val_history_.append(ndcg)
                if ndcg > max(self.val_history_[:-1], default=-1.0):
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if patience is not None and stale >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.is_fitted = True
        return self

    # -- contract: scoring ---------------------------------------------- #
    def _check(self):
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()

    def predict(self, user_id, item_id, **kwargs) -> float:
        self._check()
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        with torch.no_grad():
            q = self._query(np.array([self.user_map[user_id]]))
            items = torch.as_tensor([[self.item_map[item_id]]], device=self.device)
            return float(self.model(q, items)[0, 0])

    def recommend(self, user_id, top_k: int = 10, exclude_items=None,
                  exclude_seen: bool = True, **kwargs) -> List[Any]:
        self._check()
        if user_id not in self.user_map:
            return []
        u = self.user_map[user_id]
        with torch.no_grad():
            scores = self._score_all(self._query(np.array([u])))[0].float().cpu().numpy()
        if exclude_seen:
            lo, hi = self._seen.indptr[u], self._seen.indptr[u + 1]
            scores[self._seen.indices[lo:hi] - 1] = -np.inf
        for it in exclude_items or ():
            if it in self.item_map:
                scores[self.item_map[it] - 1] = -np.inf
        k = min(top_k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [self._items[j] for j in top]

    # -- contract: persistence ------------------------------------------ #
    def save(self, path: Union[str, Path], safe: bool = True, **kwargs) -> None:
        self._check_fitted()
        cls = self.module_cls
        if cls.__module__ == "__main__":
            warnings.warn(f"{cls.__name__} is defined in __main__: loading in another process "
                          "needs module_cls=. Put it in an importable module to avoid that.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corerec_class": "corerec.nn.recommender.Recommender",
            "module": f"{cls.__module__}:{cls.__qualname__}",
            "module_kwargs": self.module_kwargs,
            "config": {k: getattr(self, k) for k in (
                "inputs", "num_negatives", "epochs", "batch_size", "lr", "weight_decay",
                "max_len", "seed", "name")},
            "loss": self.loss if isinstance(self.loss, str) else "bpr",
            "user_ids": self._users, "item_ids": self._items,
            "seen": (self._seen.data, self._seen.indices, self._seen.indptr),
            "sequences": self._sequences,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
        }
        if safe:
            from corerec.api.bundle_helpers import pack_sparse_arrays
            from corerec.api.model_bundle import save_bundle
            state = {k: v for k, v in payload.items() if k not in ("config", "seen", "state_dict")}
            save_bundle(path, model_class=payload["corerec_class"], config=payload["config"],
                        state=state, arrays=pack_sparse_arrays({"seen": self._seen}),
                        state_dict=payload["state_dict"])
        else:
            torch.save(payload, p)

    @classmethod
    def load(cls, path: Union[str, Path], module_cls: Optional[type] = None,
             device: str = "auto", *, allow_pickle: bool = False, **kwargs) -> "Recommender":
        from corerec.api.model_bundle import is_safe_bundle, load_bundle, require_legacy_pickle
        if is_safe_bundle(path):
            from corerec.api.bundle_helpers import unpack_sparse_arrays
            bundle = load_bundle(path, map_location="cpu", allow_pickle=allow_pickle)
            seen = unpack_sparse_arrays(bundle["arrays"])["seen"]
            d = {**bundle["state"], "config": bundle["config"],
                 "seen": (seen.data, seen.indices, seen.indptr),
                 "state_dict": bundle["state_dict"]}
        else:
            require_legacy_pickle(path, allow_pickle)
            d = torch.load(Path(path), map_location="cpu", weights_only=False)
        if module_cls is None:
            mod, _, qual = d["module"].partition(":")
            from corerec.nn.models import MatrixFactorization, SequentialTransformer, HSTUTransformer
            builtins = {f"{c.__module__}:{c.__qualname__}": c
                        for c in (MatrixFactorization, SequentialTransformer, HSTUTransformer)}
            if d["module"] not in builtins and not allow_pickle:
                raise ValueError("Custom module requires module_cls= explicitly; "
                                 "artifact metadata cannot authorize importing Python modules")
            module_cls = importlib.import_module(mod)
            for part in qual.split("."):
                module_cls = getattr(module_cls, part, None)
                if module_cls is None:
                    raise ValueError(
                        f"can't find {qual} in {mod}; it was defined in a script or notebook "
                        "when saved. Pass Recommender.load(path, module_cls=...)")
        cfg = dict(d["config"])
        inst = cls(module_cls, d["module_kwargs"], loss=d["loss"], device=device, **cfg)
        inst._users, inst._items = d["user_ids"], d["item_ids"]
        inst.user_map = {x: k for k, x in enumerate(inst._users)}
        inst.item_map = {x: k + 1 for k, x in enumerate(inst._items)}
        inst.uid_map, inst.iid_map = inst.user_map, inst.item_map
        inst.num_users, inst.num_items = len(inst._users), len(inst._items)
        data, indices, indptr = d["seen"]
        inst._seen = sp.csr_matrix((data, indices, indptr),
                                   shape=(inst.num_users, inst.num_items + 1))
        inst._sequences = [np.asarray(seq, dtype=np.int64) for seq in d["sequences"]]
        inst.model = inst._build()
        inst.model.load_state_dict(d["state_dict"])
        inst.model.eval()
        inst.is_fitted = True
        return inst
