#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.dcn import DCN
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    model = DCN(embedding_dim=16, num_cross_layers=1, deep_layers=[64], epochs=1, batch_size=256, device="cpu")
    try:
        model.fit(users, items, ratings)
        recs = model.recommend(users[0], top_n=10, exclude_seen=False)
        print("DCN recommendations for", users[0], ":", recs)
    except Exception as e:
        print("DCN example skipped:", e) 