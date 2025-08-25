#!/usr/bin/env python3

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from importlib import import_module
from examples.utils_example_data import load_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    # docs keyed by item id -> text; build naive text from item id
    items = list(dict.fromkeys(data["items"]))
    docs = {it: f"content item {it} about recommendation and movies" for it in items}

    try:
        mod = import_module("corerec.engines.contentFilterEngine.tfidf_recommender")
        TFIDF = getattr(mod, "TFIDFRecommender")
    except Exception as e:
        print("TFIDFRecommender not available:", e)
        sys.exit(0)

    model = TFIDF()
    try:
        model.fit(items, docs)
        print("Query 'movie':", model.recommend_by_text("movie", top_n=10))
    except Exception as e:
        print("TFIDF example skipped:", e) 