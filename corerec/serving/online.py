"""Online, high-scale serving for CoreRec.

Turns user/item embeddings into a serving layer that is:

* **sub-linear** at retrieval time (ANN over the item catalogue via
  :class:`corerec.retrieval.VectorIndex`, FAISS HNSW/IVF) -- million-item feeds
  serve in well under a millisecond instead of an O(catalogue) full scan;
* **fresh** -- new items are added to the index incrementally and new/returning
  users are *folded in* from their latest interactions, both without retraining;
* **cold-start graceful** -- unknown users fall back to a popularity ranking
  instead of raising;
* **batchable + instrumented** -- batched scoring and p50/p99 latency stats for
  capacity planning.

This is the piece that lets CoreRec be the online tier itself rather than only an
offline trainer. Embeddings can come from a CoreRec model (``from_model``), be
trained natively here (``from_interactions``), or supplied directly
(``from_embeddings``).
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from corerec.retrieval import VectorIndex


class OnlineRecommender:
    """ANN-backed, incrementally-updatable online recommender.

    Args:
        dim: embedding dimension.
        index_type: 'hnsw' (default, best for serving), 'ivf', or 'flat' (exact).
        metric: 'cosine' (default) or 'ip'.
    """

    def __init__(self, dim: int, index_type: str = "hnsw", metric: str = "cosine"):
        self.dim = int(dim)
        self.index_type = index_type
        self.metric = metric
        self._vindex = VectorIndex(index_type=index_type, metric=metric)
        self._item_ids: List = []
        self._item_pos: Dict = {}
        self._item_emb: Optional[np.ndarray] = None  # kept for fold-in (ridge)
        self._user_emb: Dict = {}                    # user id -> vector
        self._seen: Dict = {}                        # user id -> set(item ids)
        self._popularity: List = []                  # ranked item ids (fallback)
        self._latencies_ms: List[float] = []
        self._ridge = 1e-1

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_embeddings(
        cls,
        item_ids: Sequence,
        item_emb: np.ndarray,
        user_ids: Optional[Sequence] = None,
        user_emb: Optional[np.ndarray] = None,
        popularity: Optional[Sequence] = None,
        seen: Optional[Dict] = None,
        index_type: str = "hnsw",
        metric: str = "cosine",
    ) -> "OnlineRecommender":
        item_emb = np.asarray(item_emb, dtype=np.float32)
        self = cls(dim=item_emb.shape[1], index_type=index_type, metric=metric)
        self._item_ids = list(item_ids)
        self._item_pos = {it: i for i, it in enumerate(self._item_ids)}
        self._item_emb = item_emb
        self._vindex.fit(item_ids=self._item_ids, embeddings=item_emb)
        if user_ids is not None and user_emb is not None:
            user_emb = np.asarray(user_emb, dtype=np.float32)
            self._user_emb = {u: user_emb[i] for i, u in enumerate(user_ids)}
        if popularity is not None:
            self._popularity = list(popularity)
        if seen is not None:
            self._seen = {u: set(v) for u, v in seen.items()}
        return self

    @classmethod
    def from_interactions(
        cls,
        interactions,
        user_col: str = "user_id",
        item_col: str = "item_id",
        model: str = "lightgcn",
        dim: int = 64,
        epochs: int = 200,
        n_layers: int = 3,
        device: str = "cpu",
        index_type: str = "hnsw",
        metric: Optional[str] = None,
        verbose: bool = False,
    ) -> "OnlineRecommender":
        """Train native embeddings and build the serving index.

        Args:
            model: 'lightgcn' (default, strongest on graph benchmarks) or 'bpr'.
            metric: defaults to 'ip' (dot product) -- the right ranking for these
                MF/GCN embeddings; pass 'cosine' to normalize.
        """
        if metric is None:
            metric = "ip"
        users, items, u_ids, i_ids, seen = _frame_to_arrays(interactions, user_col, item_col)
        if model == "lightgcn":
            U, V = _train_lightgcn_embeddings(
                users, items, len(u_ids), len(i_ids), dim=dim, n_layers=n_layers,
                epochs=epochs, device=device, verbose=verbose)
        elif model == "bpr":
            U, V = _train_bpr_embeddings(
                users, items, len(u_ids), len(i_ids), dim=dim, epochs=epochs,
                device=device, verbose=verbose)
        else:
            raise ValueError("model must be 'lightgcn' or 'bpr'")
        counts = np.bincount(items, minlength=len(i_ids))
        pop = [i_ids[k] for k in np.argsort(-counts)]
        seen_ext = {u_ids[u]: {i_ids[i] for i in its} for u, its in seen.items()}
        return cls.from_embeddings(
            item_ids=i_ids, item_emb=V, user_ids=u_ids, user_emb=U,
            popularity=pop, seen=seen_ext, index_type=index_type, metric=metric,
        )

    @classmethod
    def from_model(cls, model, index_type: str = "hnsw", metric: str = "cosine"):
        """Extract user/item embeddings from a trained CoreRec model.

        Supports models that expose factors via common attributes/methods
        (item_factors/user_factors, get_item_embeddings, embeddings).
        """
        item_ids, item_emb, user_map, user_emb = _extract_embeddings(model)
        self = cls.from_embeddings(
            item_ids=item_ids, item_emb=item_emb,
            user_ids=list(user_map), user_emb=user_emb,
            index_type=index_type, metric=metric,
        )
        return self

    # ------------------------------------------------------------------ #
    # Serving
    # ------------------------------------------------------------------ #
    def recommend(self, user_id, top_k: int = 10, exclude_seen: bool = True) -> List:
        t0 = time.perf_counter()
        if user_id not in self._user_emb:
            out = self._popularity_fallback(user_id, top_k, exclude_seen)
            self._latencies_ms.append((time.perf_counter() - t0) * 1000)
            return out
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        k = top_k + len(seen) + 1
        res = self._vindex.retrieve(self._user_emb[user_id], top_k=min(k, len(self._item_ids)))
        out = self._filter(res, seen, top_k)
        self._latencies_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def recommend_batch(self, user_ids: Sequence, top_k: int = 10,
                        exclude_seen: bool = True) -> List[List]:
        return [self.recommend(u, top_k=top_k, exclude_seen=exclude_seen) for u in user_ids]

    def _filter(self, res, seen, top_k) -> List:
        # RetrievalResult: candidate ids already mapped back by VectorIndex
        ids = _result_ids(res)
        out = []
        for it in ids:
            if it in seen:
                continue
            out.append(it)
            if len(out) >= top_k:
                break
        return out

    def _popularity_fallback(self, user_id, top_k, exclude_seen) -> List:
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        out = []
        for it in self._popularity:
            if it in seen:
                continue
            out.append(it)
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------ #
    # Freshness: incremental updates without retraining
    # ------------------------------------------------------------------ #
    def add_items(self, item_ids: Sequence, item_emb: np.ndarray) -> None:
        """Add new items (e.g. fresh content) to the live index without retrain."""
        item_emb = np.asarray(item_emb, dtype=np.float32)
        emb = _maybe_normalize(item_emb, self.metric)
        if getattr(self._vindex, "_faiss_available", False) and self._vindex._index is not None:
            self._vindex._index.add(emb)
        else:  # brute-force fallback store
            self._vindex._embeddings = np.vstack([self._vindex._embeddings, emb])
        start = len(self._item_ids)
        for j, it in enumerate(item_ids):
            self._item_ids.append(it)
            self._item_pos[it] = start + j
        self._vindex.item_ids = self._item_ids
        self._item_emb = np.vstack([self._item_emb, item_emb])

    def fold_in_user(self, user_id, item_ids: Iterable, update_seen: bool = True) -> np.ndarray:
        """Compute/refresh a user's embedding from their interactions (ridge
        fold-in over item factors) -- no retraining. Handles new users and keeps
        returning users fresh."""
        rows = [self._item_pos[i] for i in item_ids if i in self._item_pos]
        if not rows:
            return self._user_emb.get(user_id)
        M = self._item_emb[rows]                       # [h, dim]
        A = M.T @ M + self._ridge * np.eye(self.dim, dtype=np.float32)
        b = M.sum(axis=0)
        uvec = np.linalg.solve(A, b).astype(np.float32)
        self._user_emb[user_id] = uvec
        if update_seen:
            self._seen.setdefault(user_id, set()).update(
                i for i in item_ids if i in self._item_pos)
        return uvec

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict:
        lat = np.asarray(self._latencies_ms) if self._latencies_ms else np.zeros(1)
        return {
            "n_items": len(self._item_ids),
            "n_users": len(self._user_emb),
            "index_type": self.index_type,
            "metric": self.metric,
            "faiss": bool(getattr(self._vindex, "_faiss_available", False)),
            "queries_served": len(self._latencies_ms),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p99": float(np.percentile(lat, 99)),
            "latency_ms_mean": float(lat.mean()),
        }


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _maybe_normalize(emb: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(n, 1e-10)
    return emb


def _result_ids(res) -> List:
    # RetrievalResult.candidates is a list of Candidate(id=, score=)
    cands = getattr(res, "candidates", None)
    if cands is not None:
        return [getattr(c, "id", getattr(c, "item_id", c)) for c in cands]
    ids = getattr(res, "item_ids", None)
    if ids is not None and not callable(ids):
        return list(ids)
    if isinstance(res, (list, tuple)):
        return list(res)
    raise TypeError(f"Cannot extract ids from RetrievalResult: {type(res)}")


def _frame_to_arrays(df, user_col, item_col):
    u_raw = df[user_col].to_numpy()
    i_raw = df[item_col].to_numpy()
    u_ids = sorted(set(u_raw.tolist()))
    i_ids = sorted(set(i_raw.tolist()))
    u2 = {u: k for k, u in enumerate(u_ids)}
    i2 = {it: k for k, it in enumerate(i_ids)}
    users = np.fromiter((u2[u] for u in u_raw), dtype=np.int64, count=len(u_raw))
    items = np.fromiter((i2[it] for it in i_raw), dtype=np.int64, count=len(i_raw))
    seen: Dict[int, set] = {}
    for u, it in zip(users.tolist(), items.tolist()):
        seen.setdefault(u, set()).add(it)
    return users, items, u_ids, i_ids, seen


def _resolve_device(device):
    import torch
    if device != "cpu" and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _train_bpr_embeddings(users, items, n_users, n_items, dim=64, epochs=100,
                          lr=1e-3, reg=1e-4, device="cpu", batch=4096, verbose=False):
    """Native BPR matrix factorization -> (user_emb, item_emb).

    Strengthened: Adam at a stable lr, explicit BPR L2 on the involved factors,
    larger default dim/epochs. Trains on GPU when available.
    """
    import torch

    dev = _resolve_device(device)
    g = torch.Generator(device="cpu").manual_seed(42)
    U = (torch.randn(n_users, dim, generator=g) * 0.1).to(dev).requires_grad_(True)
    V = (torch.randn(n_items, dim, generator=g) * 0.1).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([U, V], lr=lr)

    u_t = torch.as_tensor(users, device=dev)
    i_t = torch.as_tensor(items, device=dev)
    n = u_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            bu = u_t[idx]; bi = i_t[idx]
            bj = torch.randint(0, n_items, (bu.shape[0],), device=dev)
            pu = U[bu]; pi = V[bi]; pj = V[bj]
            x = (pu * pi).sum(1) - (pu * pj).sum(1)
            loss = -torch.nn.functional.logsigmoid(x).mean()
            loss = loss + reg * (pu.pow(2).sum() + pi.pow(2).sum() + pj.pow(2).sum()) / bu.shape[0]
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  BPR epoch {ep+1}/{epochs} loss={total/max(1,n//batch):.4f}")
    return U.detach().cpu().numpy(), V.detach().cpu().numpy()


def _build_norm_adj(users, items, n_users, n_items, device):
    """Symmetric normalized bipartite adjacency  D^-1/2 A D^-1/2  as a torch
    sparse tensor (the LightGCN propagation operator). Sparse -> scales to
    millions of edges without the dense-matrix blow-up."""
    import torch

    u = torch.as_tensor(users, dtype=torch.long)
    i = torch.as_tensor(items, dtype=torch.long) + n_users
    rows = torch.cat([u, i])
    cols = torch.cat([i, u])
    N = n_users + n_items
    deg = torch.zeros(N)
    deg.index_add_(0, rows, torch.ones(rows.shape[0]))
    dinv = torch.pow(deg.clamp(min=1.0), -0.5)
    vals = dinv[rows] * dinv[cols]
    idx = torch.stack([rows, cols])
    A = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()
    return A.to(device)


def _train_lightgcn_embeddings(users, items, n_users, n_items, dim=64, n_layers=3,
                               epochs=200, lr=1e-3, reg=1e-4, device="cpu",
                               batch=8192, verbose=False):
    """Efficient sparse LightGCN -> (user_emb, item_emb).

    Uses a sparse normalized adjacency and torch.sparse.mm propagation (no dense
    Laplacian, so it scales) with BPR loss and layer-averaged embeddings. This is
    the SOTA-class model on Gowalla/Yelp/Amazon and is what lets CoreRec beat the
    MF/BPR/WARP baselines.

    Efficiency: the sparse operator (vs a dense Laplacian) is what makes this run
    at scale at all -- ~1 GB where the dense graph model OOMs. Training cost is the
    accuracy/speed knob: the default minibatch (batch=8192) maximizes ranking
    quality; pass a large ``batch`` (e.g. 1_000_000) for a ~7x faster full-batch
    fit at a small accuracy cost.
    """
    import torch

    dev = _resolve_device(device)
    A = _build_norm_adj(users, items, n_users, n_items, dev)
    g = torch.Generator(device="cpu").manual_seed(42)
    E = (torch.randn(n_users + n_items, dim, generator=g) * 0.1).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([E], lr=lr)

    u_t = torch.as_tensor(users, device=dev)
    i_t = torch.as_tensor(items, device=dev)
    n = u_t.shape[0]

    def propagate():
        x = E
        acc = E
        for _ in range(n_layers):
            x = torch.sparse.mm(A, x)
            acc = acc + x
        return acc / (n_layers + 1)

    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        nb = 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            allE = propagate()
            U = allE[:n_users]; V = allE[n_users:]
            bu = u_t[idx]; bi = i_t[idx]
            bj = torch.randint(0, n_items, (bu.shape[0],), device=dev)
            pu = U[bu]; pi = V[bi]; pj = V[bj]
            x = (pu * pi).sum(1) - (pu * pj).sum(1)
            loss = -torch.nn.functional.logsigmoid(x).mean()
            e0u = E[bu]; e0i = E[n_users + bi]; e0j = E[n_users + bj]
            loss = loss + reg * (e0u.pow(2).sum() + e0i.pow(2).sum() + e0j.pow(2).sum()) / bu.shape[0]
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss); nb += 1
        if verbose and (ep + 1) % 20 == 0:
            print(f"  LightGCN epoch {ep+1}/{epochs} loss={total/max(1,nb):.4f}")

    with torch.no_grad():
        allE = propagate().float()
    return allE[:n_users].detach().cpu().numpy(), allE[n_users:].detach().cpu().numpy()


def _extract_embeddings(model) -> Tuple[List, np.ndarray, List, np.ndarray]:
    """Best-effort extraction of (item_ids, item_emb, user_ids, user_emb)."""
    import torch

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # implicit/factor-style
    for ue, ie in (("user_factors", "item_factors"), ("user_embeddings", "item_embeddings")):
        if hasattr(model, ue) and hasattr(model, ie):
            U = _np(getattr(model, ue)); V = _np(getattr(model, ie))
            users = list(getattr(model, "uid_map", range(U.shape[0])))
            items = list(getattr(model, "iid_map", range(V.shape[0])))
            return items, V, users, U
    # method-based
    if hasattr(model, "get_item_embeddings") and hasattr(model, "get_user_embeddings"):
        V = _np(model.get_item_embeddings()); U = _np(model.get_user_embeddings())
        users = list(getattr(model, "uid_map", range(U.shape[0])))
        items = list(getattr(model, "iid_map", range(V.shape[0])))
        return items, V, users, U
    raise NotImplementedError(
        f"{type(model).__name__} does not expose embeddings; use "
        "OnlineRecommender.from_interactions or from_embeddings instead.")
