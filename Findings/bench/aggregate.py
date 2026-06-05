"""Collect results/*.json into a CSV + a markdown table for the manuscript."""
import csv
import glob
import json
import os

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

COLS = ["framework", "model", "fit_time_s", "recommend_latency_ms_per_user",
        "peak_mem_mb", "Recall@10", "NDCG@10", "HitRate@10", "RMSE", "MAE",
        "n_eval_users"]


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def main():
    rows = load()
    # CSV
    with open(os.path.join(RES, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLS})

    # Markdown (sorted by NDCG desc)
    rows.sort(key=lambda r: (r.get("NDCG@10") or -1), reverse=True)
    hdr = ["Framework", "Model", "Fit (s)", "Rec lat (ms/u)", "Peak mem (MB)",
           "Recall@10", "NDCG@10", "HitRate@10", "RMSE", "MAE"]
    md = ["| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        def fmt(x):
            return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
        md.append("| " + " | ".join([
            r["framework"], r["model"], fmt(r.get("fit_time_s")),
            fmt(r.get("recommend_latency_ms_per_user")), fmt(r.get("peak_mem_mb")),
            fmt(r.get("Recall@10")), fmt(r.get("NDCG@10")), fmt(r.get("HitRate@10")),
            fmt(r.get("RMSE")), fmt(r.get("MAE")),
        ]) + " |")
    table = "\n".join(md)
    with open(os.path.join(RES, "table.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
