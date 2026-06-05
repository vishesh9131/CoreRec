"""Aggregate same-model matrix (results/same/*.json incl. RecBole) and the LOO
study (results/loo/*.json) into CSVs + markdown grouped by model family."""
import csv
import glob
import json
import os

HERE = os.path.dirname(__file__)

# map raw model name -> comparable family
FAMILY = {
    "SAR": "Neighborhood-CF", "ItemKNN": "Neighborhood-CF", "UserKNN": "Neighborhood-CF",
    "NCF": "NeuMF/NCF", "NeuMF": "NeuMF/NCF", "GMF": "NeuMF/NCF",
    "LightGCN": "LightGCN", "GNNRec": "GNN/NGCF", "NGCF": "GNN/NGCF",
    "DCN": "DCN", "DCN_binary": "DCN", "DeepFM": "DeepFM",
    "BPR": "BPR", "MF": "MF", "SVD": "MF", "WMF": "MF",
    "VAECF": "VAE", "WARP": "WARP", "ALS": "ALS",
    "SASRec": "Sequential", "BERT4Rec": "Sequential",
}


def load(dirname):
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", dirname, "*.json"))):
        if os.path.basename(p).startswith("_"):
            continue
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def dump_csv(rows, cols, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def md_same(rows):
    for r in rows:
        r["family"] = FAMILY.get(r["model"], r["model"])
    rows.sort(key=lambda r: (r["family"], -(r.get("NDCG@10") or -1)))
    lines = ["| Family | Framework | Model | NDCG@10 | Recall@10 | HitRate@10 | Fit (s) | Mem (MB) | Device |",
             "|---|---|---|---|---|---|---|---|---|"]
    def f(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    for r in rows:
        lines.append("| " + " | ".join([
            r["family"], r["framework"], r["model"],
            f(r.get("NDCG@10")), f(r.get("Recall@10")), f(r.get("HitRate@10")),
            f(r.get("fit_time_s")), f(r.get("peak_mem_mb")), str(r.get("device", "cpu")),
        ]) + " |")
    return "\n".join(lines)


def md_loo(rows):
    for r in rows:
        r["family"] = FAMILY.get(r["model"], r["model"])
    rows.sort(key=lambda r: (r["family"], -(r.get("NDCG@10") or -1)))
    lines = ["| Family | Framework | Model | HR@10 | NDCG@10 | Fit (s) | #users |",
             "|---|---|---|---|---|---|---|"]
    def f(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    for r in rows:
        lines.append("| " + " | ".join([
            r["family"], r["framework"], r["model"],
            f(r.get("HR@10")), f(r.get("NDCG@10")), f(r.get("fit_time_s")),
            str(r.get("n_eval_users")),
        ]) + " |")
    return "\n".join(lines)


def main():
    same = load("same")
    cols = ["framework", "model", "dataset", "size", "n_users", "n_items",
            "fit_time_s", "recommend_latency_ms_per_user", "peak_mem_mb",
            "Recall@10", "NDCG@10", "HitRate@10", "device"]
    dump_csv(same, cols, os.path.join(HERE, "results", "same", "same_results.csv"))
    same_md = md_same(list(same))
    with open(os.path.join(HERE, "results", "same", "same_table.md"), "w") as f:
        f.write(same_md + "\n")
    print("=== SAME-MODEL MATRIX ===\n" + same_md + "\n")

    loo = load("loo")
    if loo:
        loo_md = md_loo(list(loo))
        with open(os.path.join(HERE, "results", "loo", "loo_table.md"), "w") as f:
            f.write(loo_md + "\n")
        print("=== LOO STUDY ===\n" + loo_md)


if __name__ == "__main__":
    main()
