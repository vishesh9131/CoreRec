#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Using new Engine-Level Organization API
from corerec.engines import unionized
from examples.utils_example_data import load_interactions, build_csr_from_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    mat, uniq_users, uniq_items = build_csr_from_interactions(users, items, ratings)
    rank = min(4, len(uniq_users), len(uniq_items))
    # New API: Direct access to RLRMC from unionized
    model = unionized.RLRMC(rank=rank, max_iter=1, verbose=False)
    model.fit(mat, uniq_users, uniq_items)
    recs = model.recommend(uniq_users[0], top_n=10)
    print("RLRMC recommendations for", uniq_users[0], ":", recs)
