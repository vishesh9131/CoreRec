#!/usr/bin/env python3
"""Generate sample_data/events.csv, the interactions file the README demo serves.

    python examples/make_demo_events.py

2,000 users and 600 items. Each user has a taste (one of 8 genres) and most of
what they watch comes from it; a share comes from a long-tailed popularity
distribution instead. That gives a model something real to learn, and a
most-popular baseline something real to lose to. Seeded, so the file is
reproducible byte for byte.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_USERS, N_ITEMS, N_GENRES = 2000, 600, 8
OUT = Path(__file__).resolve().parents[1] / "sample_data" / "events.csv"


def main(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, N_GENRES, N_ITEMS)
    popularity = 1.0 / np.arange(1, N_ITEMS + 1) ** 0.8
    popularity = popularity[rng.permutation(N_ITEMS)]
    popularity /= popularity.sum()

    rows = []
    start = 1_704_067_200  # 2024-01-01 UTC
    for user in range(N_USERS):
        genre = rng.integers(0, N_GENRES)
        in_genre = np.flatnonzero(item_genre == genre)
        p_in = popularity[in_genre] / popularity[in_genre].sum()
        n = int(rng.integers(5, 30))
        n_taste = int(round(n * 0.75))
        items = np.concatenate([
            rng.choice(in_genre, size=min(n_taste, len(in_genre)), replace=False, p=p_in),
            rng.choice(N_ITEMS, size=n - n_taste, replace=False, p=popularity),
        ])
        rng.shuffle(items)  # taste and popularity picks interleave over time
        times = np.sort(start + rng.integers(0, 180 * 86400, len(items)))
        for item, t in zip(items, times):
            liked = item_genre[item] == genre
            rating = int(rng.integers(4, 6)) if liked else int(rng.integers(1, 4))
            rows.append((f"u{user:04d}", f"i{item:03d}", rating, int(t)))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    df = df.drop_duplicates(["user_id", "item_id"]).sort_values("timestamp", kind="mergesort")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} interactions to {OUT}")
    return df


if __name__ == "__main__":
    main()
