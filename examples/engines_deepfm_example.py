#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.deepfm import DeepFM
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    model = DeepFM(embedding_dim=16, hidden_layers=[64, 32], epochs=1, batch_size=512, device="cpu")
    try:
        model.fit(users, items, ratings)
        recs = model.recommend(users[0], top_n=10, exclude_seen=False)
        print("DeepFM recommendations for", users[0], ":", recs)
    except Exception as e:
        print("DeepFM example skipped:", e) 