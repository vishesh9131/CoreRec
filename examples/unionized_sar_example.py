#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines import SAR
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users, items, ratings = data["users"], data["items"], data["ratings"]

    # SAR.fit() takes a DataFrame; fit_from_lists() takes the triple form.
    model = SAR(similarity_type="jaccard")
    model.fit_from_lists(users, items, ratings)
    recs = model.recommend(users[0], top_k=10)
    print("SAR recommendations for", users[0], ":", recs)
