#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.mind import MIND
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ts = data["users"], data["items"], data["timestamps"]

    model = MIND(embedding_dim=32, num_interests=4, epochs=1, batch_size=256, max_seq_length=50, device="cpu")
    model.fit(users, items, ts)
    recs = model.recommend(users[0], top_n=10)
    print("MIND recommendations for", users[0], ":", recs) 