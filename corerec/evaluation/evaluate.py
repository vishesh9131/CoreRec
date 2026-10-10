"""One-call evaluation for any CoreRec recommender.

Mirrors what Cornac/RecBole offer out of the box: hand it a fitted model and a
test set, get back the standard top-K ranking metrics, computed against the
model's own ``recommend`` path (i.e. the real serving path).
"""
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from corerec.evaluation.metrics import RankingMetrics, _validate_k

logger = logging.getLogger(__name__)


# canonical names, as in every report and the CLI
METRICS = {
    "NDCG": RankingMetrics.ndcg_at_k,
    "MAP": RankingMetrics.map_at_k,
    "MRR": RankingMetrics.mrr_at_k,
    "Precision": RankingMetrics.precision_at_k,
    "Recall": RankingMetrics.recall_at_k,
    "HitRate": RankingMetrics.hit_rate_at_k,
}


def _to_triples(data, user_col, item_col, rating_col):
    """Normalize a DataFrame or iterable of tuples into (user, item, rating)."""
    if hasattr(data, "columns"):  # pandas DataFrame
        users = data[user_col].tolist()
        items = data[item_col].tolist()
        if rating_col in data.columns:
            ratings = data[rating_col].tolist()
        else:
            ratings = [1.0] * len(users)
        return list(zip(users, items, ratings))
    triples = []
    for row in data:
        if len(row) >= 3:
            triples.append((row[0], row[1], row[2]))
        else:
            triples.append((row[0], row[1], 1.0))
    return triples


def evaluate(
    model: Any,
    test_interactions: Iterable,
    train_interactions: Optional[Iterable] = None,
    k: Union[int, Iterable[int]] = 10,
    relevance_threshold: Optional[float] = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
    user_subset: Optional[Iterable] = None,
    verbose: bool = False,
    strict: bool = False,
) -> Dict[str, float]:
    """Evaluate a fitted recommender on held-out interactions.

    Args:
        model: a fitted CoreRec model exposing ``recommend(user_id, top_k=...)``.
        test_interactions: DataFrame or iterable of (user, item[, rating]).
        train_interactions: optional; items here are excluded from each user's
            recommendations (standard "exclude seen" protocol).
        k: cutoff for the @K metrics, or several (``k=[10, 50]``), all scored
            from one recommend() call per user.
        relevance_threshold: if set, a test item counts as relevant only when its
            rating >= threshold; otherwise every test item is relevant.
        user_subset: optional iterable of user ids to restrict evaluation to.
        strict: re-raise a user's recommend() error instead of counting it in n_errors.

    Returns:
        dict with NDCG@k, MAP@k, MRR@k, Precision@k, Recall@k, HitRate@k and the
        number of users evaluated.
    """
    try:
        ks = [k] if np.isscalar(k) else list(k)
    except TypeError as exc:
        raise ValueError("k must be a positive integer or a nonempty iterable of cutoffs") from exc
    if not ks:
        raise ValueError("k must contain at least one cutoff")
    for cutoff in ks:
        _validate_k(cutoff)
    ks = sorted(set(ks))
    k_max = ks[-1]

    test = _to_triples(test_interactions, user_col, item_col, rating_col)

    # ground truth: relevant items per user
    relevant = defaultdict(set)
    for u, it, r in test:
        if relevance_threshold is None or r >= relevance_threshold:
            relevant[u].add(it)

    seen = defaultdict(set)
    if train_interactions is not None:
        for u, it, _ in _to_triples(train_interactions, user_col, item_col, rating_col):
            seen[u].add(it)

    users = list(relevant.keys()) if user_subset is None else list(user_subset)

    scores = defaultdict(list)
    n_eval = n_errors = n_repeats = 0
    for u in users:
        truth = relevant.get(u)
        if not truth:
            continue
        # request extra to compensate for seen-item exclusion, then trim
        try:
            recs = model.recommend(u, top_k=k_max + len(seen.get(u, ())))
        except Exception:
            if strict:
                raise
            # usually a user the model never saw; counted so it isn't invisible
            n_errors += 1
            continue
        if recs is None:
            continue
        recs = [r[0] if isinstance(r, (tuple, list)) else r for r in recs]
        if u in seen:
            recs = [it for it in recs if it not in seen[u]]
        recs = recs[:k_max]
        if len(set(recs)) < len(recs):
            n_repeats += 1  # scored as misses by RankingMetrics
        truth = list(truth)

        for kk in ks:  # each metric reads only recs[:kk]
            for name, fn in METRICS.items():
                scores[f"{name}@{kk}"].append(fn(recs, truth, kk))
        n_eval += 1

    results = {f"{name}@{kk}": float(np.mean(scores[f"{name}@{kk}"])) if n_eval else 0.0
               for kk in ks for name in METRICS}
    results.update(n_users=n_eval, n_errors=n_errors)
    if n_repeats:
        logger.warning("%s returned the same item twice in one list for %d of %d users; "
                       "repeats count as misses", type(model).__name__, n_repeats, n_eval)
    if verbose:
        for key, val in results.items():
            print(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")
    return results
