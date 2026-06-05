"""Identical evaluation for every framework.

All frameworks are scored by THIS code, not their own evaluators, so metric
definitions are constant across the comparison.

Ranking protocol: for each test user with >=1 relevant item, score every item,
mask items seen in train (set to -inf), take the top-K, and compute Recall@K,
NDCG@K, HitRate@K against that user's relevant set.
"""
import numpy as np


def rmse(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true)))


def _dcg(hits):
    # hits: binary array in rank order
    return np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))


def loo_metrics(score_fn, candidates, k=10, user_subset=None):
    """Leave-one-out + sampled negatives (He et al. NeuMF protocol).

    candidates: uidx -> (pos_iidx, np.array negatives). For each user we rank the
    1 positive among the n_neg negatives and compute HR@K and NDCG@K.
    """
    users = list(candidates.keys()) if user_subset is None else user_subset
    hrs, ndcgs = [], []
    rng = np.random.RandomState(0)
    for u in users:
        pos, negs = candidates[u]
        items = np.concatenate(([pos], negs))
        # shuffle so that tied scores (e.g. a collapsed constant-output model)
        # do not break in favour of the positive, which would inflate metrics
        perm = rng.permutation(len(items))
        items_s = items[perm]
        pos_pos = int(np.where(perm == 0)[0][0])  # where the positive landed
        scores = np.asarray(score_fn(u), dtype=float)[items_s]
        order = np.argsort(-scores)
        rank = int(np.where(order == pos_pos)[0][0])  # position of the positive
        if rank < k:
            hrs.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            hrs.append(0.0)
            ndcgs.append(0.0)
    return {f"HR@{k}": float(np.mean(hrs)) if hrs else 0.0,
            f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "n_eval_users": len(hrs)}


def ranking_metrics(score_fn, n_items, seen, relevant, k=10, user_subset=None):
    """score_fn(uidx) -> np.ndarray[n_items] of recommendation scores (higher=better).

    Returns dict of mean Recall@K, NDCG@K, HitRate@K over evaluated users.
    """
    users = list(relevant.keys()) if user_subset is None else user_subset
    recalls, ndcgs, hrs = [], [], []
    for u in users:
        rel = relevant.get(u)
        if not rel:
            continue
        scores = np.asarray(score_fn(u), dtype=float).copy()
        if scores.shape[0] != n_items:
            raise ValueError(f"score_fn returned {scores.shape[0]} != {n_items}")
        for i in seen.get(u, ()):  # mask training items
            scores[i] = -np.inf
        topk = np.argpartition(-scores, min(k, n_items - 1))[:k]
        topk = topk[np.argsort(-scores[topk])]
        hits = np.array([1.0 if i in rel else 0.0 for i in topk])

        n_rel = len(rel)
        recalls.append(hits.sum() / n_rel)
        idcg = _dcg(np.ones(min(n_rel, k)))
        ndcgs.append(_dcg(hits) / idcg if idcg > 0 else 0.0)
        hrs.append(1.0 if hits.sum() > 0 else 0.0)

    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hrs)) if hrs else 0.0,
        "n_eval_users": len(recalls),
    }
