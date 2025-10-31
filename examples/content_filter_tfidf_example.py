#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Using new Engine-Level Organization API
from corerec.engines import content
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    # docs keyed by item id -> text; build naive text from item id
    items = list(dict.fromkeys(data["items"]))
    docs = {it: f"content item {it} about recommendation and movies" for it in items}

    # New API: Direct access to TFIDFRecommender from content
    try:
        model = content.TFIDFRecommender()
    except Exception as e:
        print("TFIDFRecommender not available:", e)
        sys.exit(0)
    try:
        model.fit(items, docs)
        print("Query 'movie':", model.recommend_by_text("movie", top_n=10))
    except Exception as e:
        print("TFIDF example skipped:", e) 