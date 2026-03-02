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

    # New API: Access through unionized namespace
    # Note: FAST class might be named differently, trying common variations
    try:
        model = unionized.FAST(factors=32, iterations=1, batch_size=1024, seed=42)
    except AttributeError:
        # Fallback to FastRecommender if FAST doesn't exist
        try:
            from corerec.engines.collaborative.fast import FAST

            model = FAST(factors=32, iterations=1, batch_size=1024, seed=42)
        except:
            print("FAST model not available")
            sys.exit(0)
    model.fit(users, items, ratings)
    recs = model.recommend(users[0], top_n=10)
    print("FAST recommendations for", users[0], ":", recs)
