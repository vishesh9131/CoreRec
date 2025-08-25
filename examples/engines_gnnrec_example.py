#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.gnnrec import GNNRec
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    model = GNNRec(embedding_dim=32, num_gnn_layers=1, epochs=1, batch_size=1024, device="cpu")
    model.fit(users, items, ratings)
    recs = model.recommend(users[0], top_n=10, exclude_seen=True)
    print("GNNRec recommendations for", users[0], ":", recs) 