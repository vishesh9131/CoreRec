#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Using new Engine-Level Organization API
from corerec import engines
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    # New API: Direct access to NASRec from engines
    model = engines.NASRec(embedding_dim=32, hidden_dims=[64, 32], epochs=1, batch_size=256, device="cpu")
    model.fit(users, items, ratings)
    recs = model.recommend(users[0], top_n=10, exclude_seen=False)
    print("NASRec recommendations for", users[0], ":", recs)
