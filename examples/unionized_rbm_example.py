#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.unionizedFilterEngine.rbm import RBM
from examples.utils_example_data import load_interactions, build_csr_from_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    mat, uniq_users, uniq_items = build_csr_from_interactions(users, items, ratings)
    model = RBM(n_hidden=64, n_epochs=1, batch_size=64, verbose=True)
    try:
        model.fit(mat, uniq_users, uniq_items)
        recs = model.recommend(uniq_users[0], top_n=10)
        print("RBM recommendations for", uniq_users[0], ":", recs)
    except Exception as e:
        print("RBM example skipped:", e) 