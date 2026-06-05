"""Aggregate the scalability sweep (results/scale/*.json) into tidy CSV and
per-metric pivot tables (rows=model, cols=interaction count)."""
import csv
import glob
import json
import os

HERE = os.path.dirname(__file__)
SCALE = os.path.join(HERE, "results", "scale")


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(SCALE, "*.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def pivot(rows, metric):
    sizes = sorted({r["size"] for r in rows})
    keys = sorted({(r["framework"], r["model"]) for r in rows})
    lines = ["| Framework/Model | " + " | ".join(f"{s:,}" for s in sizes) + " |",
             "|" + "|".join(["---"] * (len(sizes) + 1)) + "|"]
    table = {}
    for r in rows:
        table[(r["framework"], r["model"], r["size"])] = r.get(metric)
    for fw, md in keys:
        cells = []
        for s in sizes:
            v = table.get((fw, md, s))
            cells.append("-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {fw} {md} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    rows = load()
    cols = ["framework", "model", "dataset", "size", "n_users", "n_items",
            "fit_time_s", "recommend_latency_ms_per_user", "peak_mem_mb",
            "Recall@10", "NDCG@10", "HitRate@10"]
    with open(os.path.join(SCALE, "scale_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    out = []
    for metric in ["NDCG@10", "Recall@10", "fit_time_s",
                   "recommend_latency_ms_per_user", "peak_mem_mb"]:
        out.append(f"### {metric} by scale\n")
        out.append(pivot(rows, metric))
        out.append("")
    md = "\n".join(out)
    with open(os.path.join(SCALE, "scale_tables.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
