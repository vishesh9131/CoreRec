#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Using new Engine-Level Organization API
from corerec.engines import unionized
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    # New API: Direct access to GeoMLC from unionized
    model = unionized.GeoMLC(n_factors=16, n_epochs=1, batch_size=256, device="cpu")
    model.fit(users, items, ratings)
    recs = model.recommend(users[0], top_n=10)
    print("GeoMLC recommendations for", users[0], ":", recs)
