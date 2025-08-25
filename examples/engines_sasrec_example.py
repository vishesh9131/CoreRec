#!/usr/bin/env python3

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.sasrec import SASRec
from examples.utils_example_data import load_interactions, build_csr_from_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items = data["users"], data["items"]

    mat, uniq_users, uniq_items = build_csr_from_interactions(users, items)
    model = SASRec(hidden_units=64, num_blocks=1, num_heads=1, num_epochs=1, batch_size=256, max_seq_length=50, device="cpu")
    model.fit(mat, uniq_users, uniq_items)
    recs = model.recommend(uniq_users[0], top_n=10, exclude_seen=True)
    print("SASRec recommendations for", uniq_users[0], ":", recs) 