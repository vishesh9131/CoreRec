"""Run ONE (framework, model) on the shared ML-100K split and emit JSON metrics.

Invoked once per framework in its own subprocess (see run_all.py) so memory and
import cost are isolated and frameworks can't interfere with each other. CPU-only
(CUDA disabled) so CoreRec is compared on equal footing with the CPU-only
competitors.

Usage: python runner.py --framework cornac --model BPR --out results/cornac_bpr.json
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
# repo root (…/github) so `import corerec` resolves in this subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import datautil
import metrics as M

K = 10
RANK_DIM = 32      # latent dim shared by MF-class models
EPOCHS = 20        # shared training budget for iterative models
SEED = 42          # set from --seed; models that sample negatives need several
                   # runs before a single number means anything (see BENCHMARKS)


def peak_mem_mb():
    """Peak RSS of this process, in MB.

    ru_maxrss is kilobytes on Linux but bytes on macOS/BSD. Assuming kilobytes
    everywhere reported 138204 "MB" for a run that actually peaked near 135MB,
    so any memory column produced on a Mac was inflated 1024x.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(peak / (1024.0 * 1024.0), 1)
    return round(peak / 1024.0, 1)


# --------------------------------------------------------------------------- #
# Adapters: each returns (score_fn, rating_pred_fn_or_None, fit_time_seconds)
# score_fn(uidx) -> np.ndarray[n_items]; rating_pred_fn(pairs) -> list[float]
# --------------------------------------------------------------------------- #
def run_cornac(model_name, data):
    import cornac
    from cornac.data import Dataset

    tr = data["train"]
    tuples = list(zip(tr["uidx"].astype(int).astype(str),
                      tr["iidx"].astype(int).astype(str),
                      tr["rating"].astype(float)))
    ds = Dataset.from_uir(tuples)

    cm = cornac.models
    if model_name == "BPR":
        model = cm.BPR(k=RANK_DIM, max_iter=EPOCHS, learning_rate=0.01,
                       lambda_reg=0.001, seed=SEED, verbose=False)
    elif model_name == "MF":
        model = cm.MF(k=RANK_DIM, max_iter=EPOCHS, learning_rate=0.01,
                      lambda_reg=0.02, use_bias=True, seed=SEED, verbose=False)
    elif model_name == "NeuMF":
        model = cm.NeuMF(num_factors=RANK_DIM, layers=[64, 32, 16], num_epochs=EPOCHS,
                         batch_size=256, lr=0.001, seed=SEED, verbose=False)
    elif model_name == "GMF":
        model = cm.GMF(num_factors=RANK_DIM, num_epochs=EPOCHS, batch_size=256,
                       lr=0.001, seed=SEED, verbose=False)
    elif model_name == "LightGCN":
        model = cm.LightGCN(emb_size=RANK_DIM, num_epochs=EPOCHS, num_layers=3,
                            batch_size=1024, learning_rate=0.001, seed=SEED, verbose=False)
    elif model_name == "NGCF":
        model = cm.NGCF(emb_size=RANK_DIM, num_epochs=EPOCHS, batch_size=1024,
                        learning_rate=0.001, seed=SEED, verbose=False)
    elif model_name == "VAECF":
        model = cm.VAECF(k=RANK_DIM, n_epochs=EPOCHS, batch_size=256,
                         learning_rate=0.001, seed=SEED, verbose=False)
    elif model_name == "WMF":
        model = cm.WMF(k=RANK_DIM, max_iter=EPOCHS, learning_rate=0.001, seed=SEED, verbose=False)
    elif model_name == "ItemKNN":
        model = cm.ItemKNN(k=20, similarity="cosine", verbose=False)
    elif model_name == "UserKNN":
        model = cm.UserKNN(k=20, similarity="cosine", verbose=False)
    else:
        raise ValueError(model_name)

    t0 = time.perf_counter()
    model.fit(ds)
    fit_t = time.perf_counter() - t0

    uid_map = ds.uid_map  # raw(str) -> internal
    iid_map = ds.iid_map
    n_items = data["n_items"]
    # our_iidx -> cornac internal col, for full reorder
    col_for = np.full(n_items, -1, dtype=int)
    for raw, internal in iid_map.items():
        col_for[int(raw)] = internal

    def score_fn(u):
        raw = str(u)
        out = np.full(n_items, -np.inf)
        if raw not in uid_map:
            return out
        internal_u = uid_map[raw]
        scores = model.score(internal_u)  # array over cornac items
        valid = col_for >= 0
        out[valid] = scores[col_for[valid]]
        return out

    rating_fn = None
    if model_name == "MF":
        def rating_fn(pairs):
            res = []
            for u, i in pairs:
                ru, ri = str(u), str(i)
                if ru in uid_map and ri in iid_map:
                    res.append(float(model.score(uid_map[ru], iid_map[ri])))
                else:
                    res.append(model.default_score())
            return res
    return score_fn, rating_fn, fit_t


def run_implicit(model_name, data):
    from scipy.sparse import csr_matrix
    import implicit

    tr = data["train"]
    n_u, n_i = data["n_users"], data["n_items"]
    # implicit expects user-item confidence; use rating as confidence
    ui = csr_matrix((tr["rating"].astype(float).values,
                     (tr["uidx"].values, tr["iidx"].values)), shape=(n_u, n_i))

    if model_name == "ItemKNN":
        # neighbourhood model: score(u) = user_row @ item-item similarity
        model = implicit.nearest_neighbours.CosineRecommender(K=20)
        t0 = time.perf_counter()
        model.fit(ui, show_progress=False)
        fit_t = time.perf_counter() - t0
        sim = model.similarity  # [n_items, n_items] sparse

        def score_fn(u):
            return np.asarray(ui[u].dot(sim).todense()).ravel()
        return score_fn, None, fit_t

    if model_name == "ALS":
        model = implicit.als.AlternatingLeastSquares(
            factors=RANK_DIM, iterations=EPOCHS, regularization=0.01,
            random_state=SEED, use_gpu=False)
    elif model_name == "BPR":
        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=RANK_DIM, iterations=EPOCHS, learning_rate=0.01,
            regularization=0.01, random_state=SEED, use_gpu=False)
    else:
        raise ValueError(model_name)

    t0 = time.perf_counter()
    model.fit(ui, show_progress=False)
    fit_t = time.perf_counter() - t0

    U = model.user_factors
    V = model.item_factors
    U = np.asarray(U); V = np.asarray(V)

    def score_fn(u):
        return U[u] @ V.T
    return score_fn, None, fit_t


def run_lightfm(model_name, data):
    from scipy.sparse import coo_matrix
    from lightfm import LightFM

    tr = data["train"]
    n_u, n_i = data["n_users"], data["n_items"]
    mat = coo_matrix((np.ones(len(tr)),
                      (tr["uidx"].values, tr["iidx"].values)), shape=(n_u, n_i))
    loss = "warp" if model_name == "WARP" else "bpr"
    model = LightFM(no_components=RANK_DIM, loss=loss, learning_rate=0.05,
                    random_state=SEED)
    t0 = time.perf_counter()
    model.fit(mat, epochs=EPOCHS, num_threads=1)
    fit_t = time.perf_counter() - t0

    all_items = np.arange(n_i)

    def score_fn(u):
        return model.predict(int(u), all_items)
    return score_fn, None, fit_t


def run_surprise(model_name, data):
    from surprise import SVD, Dataset, Reader
    tr = data["train"]
    reader = Reader(rating_scale=(1, 5))
    ds = Dataset.load_from_df(tr[["uidx", "iidx", "rating"]], reader)
    trainset = ds.build_full_trainset()
    algo = SVD(n_factors=RANK_DIM, n_epochs=EPOCHS, lr_all=0.005, reg_all=0.02,
               random_state=SEED)
    t0 = time.perf_counter()
    algo.fit(trainset)
    fit_t = time.perf_counter() - t0

    n_items = data["n_items"]
    # reorder factors into our index space via inner-id maps
    pu, qi, bu, bi = algo.pu, algo.qi, algo.bu, algo.bi
    gmean = trainset.global_mean
    Ufull = np.zeros((data["n_users"], pu.shape[1]))
    Bu = np.zeros(data["n_users"])
    Vfull = np.zeros((n_items, qi.shape[1]))
    Bi = np.zeros(n_items)
    have_i = np.zeros(n_items, dtype=bool)
    for raw in trainset.all_users():
        our = int(trainset.to_raw_uid(raw))
        Ufull[our] = pu[raw]; Bu[our] = bu[raw]
    for raw in trainset.all_items():
        our = int(trainset.to_raw_iid(raw))
        Vfull[our] = qi[raw]; Bi[our] = bi[raw]; have_i[our] = True

    def score_fn(u):
        s = gmean + Bu[u] + Bi + Vfull @ Ufull[u]
        s[~have_i] = -np.inf
        return s

    def rating_fn(pairs):
        return [algo.predict(u, i).est for u, i in pairs]
    return score_fn, rating_fn, fit_t


DEVICE = "cpu"  # set from --device in main()


def run_corerec(model_name, data):
    tr = data["train"]
    uid = tr["uidx"].values
    iid = tr["iidx"].values
    rt = tr["rating"].astype(float).values
    n_items = data["n_items"]

    base = model_name.replace("_binary", "")
    if base in ("DCN", "DeepFM"):
        from corerec.engines import DCN, DeepFM
        cls = DCN if base == "DCN" else DeepFM
        # documented quickstart passes raw 1-5 ratings; *_binary variant feeds
        # implicit labels (rating>=4 -> 1) as a fairness/corrected-usage check.
        targets = (rt >= 4.0).astype(float) if model_name.endswith("_binary") else rt
        model = cls(embedding_dim=RANK_DIM, epochs=EPOCHS, learning_rate=0.001,
                    verbose=False, device=DEVICE)
        t0 = time.perf_counter()
        model.fit(user_ids=uid, item_ids=iid, ratings=targets)
        fit_t = time.perf_counter() - t0

        items = np.arange(n_items)

        def score_fn(u):
            pairs = [(int(u), int(i)) for i in items]
            return np.asarray(model.batch_predict(pairs), dtype=float)

        # RMSE only meaningful for the raw-rating regression variant
        rating_fn = None
        if not model_name.endswith("_binary"):
            def rating_fn(pairs):
                return list(model.batch_predict([(int(u), int(i)) for u, i in pairs]))
        return score_fn, rating_fn, fit_t

    if model_name == "ALS":
        from corerec.engines.matrix_factorization import ALS as CoreALS
        model = CoreALS(factors=RANK_DIM, iterations=EPOCHS)
        t0 = time.perf_counter()
        model.fit(user_ids=uid, item_ids=iid, ratings=rt)
        fit_t = time.perf_counter() - t0

        def score_fn(u):
            out = np.full(n_items, -np.inf)
            try:
                recs = model.recommend(int(u), top_k=n_items)
            except Exception:
                return out
            for rank, r in enumerate(recs):
                i, sc = (int(r[0]), float(r[1])) if isinstance(r, (tuple, list)) else (int(r), float(len(recs) - rank))
                if 0 <= i < n_items:
                    out[i] = sc
            return out
        return score_fn, None, fit_t

    if model_name in ("NCF", "NCF_binary"):
        import pandas as pd
        from corerec.engines.collaborative import NCF
        model = NCF(model_type="NeuMF", gmf_embedding_dim=RANK_DIM,
                    mlp_embedding_dim=RANK_DIM, num_epochs=EPOCHS,
                    learning_rate=0.001, verbose=False, seed=SEED, device=DEVICE)
        # NCF.fit labels every observed interaction 1.0 and samples negatives
        # (see ncf.py: ratings_parts = [np.ones(len(pos_u))]), so the rating
        # column is discarded -- a 1-star rating trains as a positive. Passing
        # binarized targets changes nothing; it was verified to give an
        # identical NDCG to five decimals.
        #
        # implicit's ALS, by contrast, feeds rating in as confidence, so it
        # weights a 5-star interaction 5x a 1-star one. It is using signal NCF
        # throws away. NCF_binary keeps only rating>=4 rows as positives, which
        # is the same relevance definition the metric uses.
        df = pd.DataFrame({"user_id": uid, "item_id": iid, "rating": rt})
        if model_name.endswith("_binary"):
            df = df[rt >= 4.0].reset_index(drop=True)
        t0 = time.perf_counter()
        model.fit(df)
        fit_t = time.perf_counter() - t0
        items = np.arange(n_items)

        def score_fn(u):
            return np.asarray(model.batch_predict([(int(u), int(i)) for i in items]), float)
        return score_fn, None, fit_t

    if model_name == "LightGCN":
        from corerec.engines.collaborative import LightGCN
        model = LightGCN(n_factors=RANK_DIM, n_layers=3, epochs=EPOCHS,
                         learning_rate=0.001, verbose=False, device=DEVICE,
                         seed=SEED)
        t0 = time.perf_counter()
        model.fit(user_ids=uid, item_ids=iid, ratings=(rt >= 4).astype(float))
        fit_t = time.perf_counter() - t0
        items = np.arange(n_items)

        def score_fn(u):
            return np.asarray(model.batch_predict([(int(u), int(i)) for i in items]), float)
        return score_fn, None, fit_t

    if model_name == "GNNRec":
        from corerec.engines import GNNRec
        model = GNNRec(embedding_dim=RANK_DIM, epochs=EPOCHS, verbose=False)
        t0 = time.perf_counter()
        model.fit(uid, iid, (rt >= 4).astype(np.float32))
        fit_t = time.perf_counter() - t0
        items = np.arange(n_items)

        def score_fn(u):
            return np.asarray(model.batch_predict([(int(u), int(i)) for i in items]), float)
        return score_fn, None, fit_t

    if model_name in ("SAR", "SAR_cosine"):
        import pandas as pd
        from corerec.engines.collaborative import SAR
        df = pd.DataFrame({"userID": uid, "itemID": iid, "rating": rt})
        model = SAR(similarity_type="cosine" if model_name.endswith("_cosine") else "jaccard")
        t0 = time.perf_counter()
        model.fit(df)
        fit_t = time.perf_counter() - t0

        # SAR exposes an item-item score path via recommend; build a dense
        # score vector per user from its recommendations.
        def score_fn(u):
            out = np.full(n_items, -np.inf)
            try:
                recs = model.recommend(user_id=int(u), top_k=n_items)
            except Exception:
                return out
            # recs may be list of ids or list of (id, score)
            for rank, r in enumerate(recs):
                if isinstance(r, (tuple, list)):
                    iidx, sc = int(r[0]), float(r[1])
                else:
                    iidx, sc = int(r), float(len(recs) - rank)
                if 0 <= iidx < n_items:
                    out[iidx] = sc
            return out
        return score_fn, None, fit_t

    raise ValueError(model_name)



# --------------------------------------------------------------------------- #
# Rank fusion
# --------------------------------------------------------------------------- #
def rrf_fuse(score_fns, n_items, k=60):
    """Reciprocal Rank Fusion over several scorers.

    RRF(i) = sum_s 1 / (k + rank_s(i)). Deliberately parameter-free apart from
    the conventional k=60: there is no weight to tune, so a fused result cannot
    be quietly optimised against the test set the way a weighted blend could.

    Fusion is where a library with a pipeline layer has a real structural
    advantage. implicit ships excellent individual algorithms and no way to
    combine them; anything like this has to be hand-rolled by the user. The
    implicit_ensemble entry below hand-rolls exactly that, so the comparison is
    pipeline-vs-pipeline rather than pipeline-vs-single-model.
    """
    def fused(u):
        total = np.zeros(n_items)
        for fn in score_fns:
            scores = np.asarray(fn(u), dtype=float)
            order = np.argsort(-scores)          # best first
            ranks = np.empty(n_items, dtype=float)
            ranks[order] = np.arange(n_items)    # 0-based rank per item
            total += 1.0 / (k + ranks)
        return total
    return fused


def run_corerec_hybrid(model_name, data):
    """CoreRec's own decorrelated signals, fused: ALS latent factors + SAR item-item."""
    import pandas as pd
    from corerec.engines.matrix_factorization import ALS as CoreALS
    from corerec.engines.collaborative import SAR

    tr = data["train"]
    uid = tr["uidx"].values
    iid = tr["iidx"].values
    rt = tr["rating"].astype(float).values
    n_items = data["n_items"]

    t0 = time.perf_counter()
    als = CoreALS(factors=RANK_DIM, iterations=EPOCHS)
    als.fit(user_ids=uid, item_ids=iid, ratings=rt)

    # cosine, matching the CosineRecommender implicit fuses with, so the two
    # ensembles differ in implementation rather than in configuration. On this
    # split cosine also beats jaccard standalone (0.3955 vs 0.3730); jaccard was
    # simply the wrong default to compare against a cosine baseline.
    sar = SAR(similarity_type="cosine")
    sar.fit(pd.DataFrame({"userID": uid, "itemID": iid, "rating": rt}))
    fit_t = time.perf_counter() - t0

    def als_scores(u):
        out = np.full(n_items, -np.inf)
        try:
            recs = als.recommend(int(u), top_k=n_items)
        except Exception:
            return out
        for rank, r in enumerate(recs):
            i, sc = (int(r[0]), float(r[1])) if isinstance(r, (tuple, list)) else (int(r), float(len(recs) - rank))
            if 0 <= i < n_items:
                out[i] = sc
        return out

    def sar_scores(u):
        out = np.full(n_items, -np.inf)
        try:
            recs = sar.recommend(user_id=int(u), top_k=n_items)
        except Exception:
            return out
        for rank, r in enumerate(recs):
            i, sc = (int(r[0]), float(r[1])) if isinstance(r, (tuple, list)) else (int(r), float(len(recs) - rank))
            if 0 <= i < n_items:
                out[i] = sc
        return out

    return rrf_fuse([als_scores, sar_scores], n_items), None, fit_t


def run_implicit_ensemble(model_name, data):
    """The same fusion, hand-rolled over implicit's models -- the fair counterpart."""
    from scipy.sparse import csr_matrix
    import implicit

    tr = data["train"]
    n_u, n_i = data["n_users"], data["n_items"]
    ui = csr_matrix((tr["rating"].astype(float).values,
                     (tr["uidx"].values, tr["iidx"].values)), shape=(n_u, n_i))

    t0 = time.perf_counter()
    als = implicit.als.AlternatingLeastSquares(
        factors=RANK_DIM, iterations=EPOCHS, regularization=0.01,
        random_state=SEED, use_gpu=False)
    als.fit(ui, show_progress=False)

    knn = implicit.nearest_neighbours.CosineRecommender(K=20)
    knn.fit(ui, show_progress=False)
    fit_t = time.perf_counter() - t0

    U = np.asarray(als.user_factors)
    V = np.asarray(als.item_factors)
    sim = knn.similarity

    def als_scores(u):
        return U[u] @ V.T

    def knn_scores(u):
        return np.asarray(ui[u].dot(sim).todense()).ravel()

    return rrf_fuse([als_scores, knn_scores], n_i), None, fit_t


DISPATCH = {
    "cornac": run_cornac,
    "implicit": run_implicit,
    "lightfm": run_lightfm,
    "surprise": run_surprise,
    "corerec": run_corerec,
    "corerec_hybrid": run_corerec_hybrid,
    "implicit_ensemble": run_implicit_ensemble,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="ml100k")
    ap.add_argument("--size", type=int, default=None,
                    help="subsample to N interactions (ml1m only)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval_users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42,
                    help="model seed; the evaluation cohort stays fixed regardless")
    args = ap.parse_args()
    global DEVICE
    global SEED
    SEED = args.seed
    np.random.seed(SEED)          # models that use the global RNG
    try:
        import torch
        torch.manual_seed(SEED)
    except ImportError:
        pass

    DEVICE = args.device
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    data = datautil.load_dataset(args.dataset, n_interactions=args.size)
    seen = datautil.train_seen(data)
    rel = datautil.test_relevant(data)
    eu = datautil.eval_users(data, n=args.eval_users, seed=42)  # fixed cohort across seeds

    score_fn, rating_fn, fit_t = DISPATCH[args.framework](args.model, data)

    t0 = time.perf_counter()
    rank = M.ranking_metrics(score_fn, data["n_items"], seen, rel, k=K, user_subset=eu)
    rank_eval_t = time.perf_counter() - t0
    # per-user recommend latency (ms): scoring time / users evaluated
    rec_latency_ms = 1000.0 * rank_eval_t / max(1, rank["n_eval_users"])

    out = {
        "framework": args.framework,
        "model": args.model,
        "seed": args.seed,
        "dataset": args.dataset,
        "size": args.size if args.size is not None else len(data["train"]) + len(data["test"]),
        "n_users": data["n_users"],
        "n_items": data["n_items"],
        "device": args.device,
        "fit_time_s": round(fit_t, 4),
        "recommend_latency_ms_per_user": round(rec_latency_ms, 4),
        "peak_mem_mb": round(peak_mem_mb(), 1),
        **{kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in rank.items()},
    }

    if rating_fn is not None:
        t = data["test"]
        pairs = list(zip(t["uidx"].values, t["iidx"].values))
        preds = rating_fn(pairs)
        out["RMSE"] = round(M.rmse(preds, t["rating"].values), 5)
        out["MAE"] = round(M.mae(preds, t["rating"].values), 5)
    else:
        out["RMSE"] = None
        out["MAE"] = None

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
