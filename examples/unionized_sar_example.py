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

    # New API: Direct access to SAR from unionized
    model = unionized.SAR(similarity_type="jaccard")
    model.fit(users, items, ratings)
    recs = model.recommend(users[0], top_n=10, exclude_seen=True)
    print("SAR recommendations for", users[0], ":", recs)
