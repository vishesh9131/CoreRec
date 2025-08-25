#!/usr/bin/env python3
# quickstart for contentFilterEngine

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from importlib import import_module


def run_tfidf():
    try:
        # import tfidf directly by path through package
        mod = import_module("corerec.engines.contentFilterEngine.tfidf_recommender")
        TFIDF = getattr(mod, "TFIDFRecommender")
    except Exception as e:
        print("TFIDFRecommender not available:", e)
        return

    docs = {
        1: "machine learning recommender systems",
        2: "deep learning for recommendation",
        3: "graph neural networks",
        4: "content based filtering with tfidf",
    }
    users = [100, 100, 101]
    items = [1, 2, 3]
    ratings = [1.0, 1.0, 1.0]

    model = TFIDF()
    try:
        model.fit(items, docs)  # minimal index
        print("TFIDF recs for text 'recommender':", model.recommend_by_text("recommender", top_n=3))
    except Exception as e:
        print("TFIDF run skipped:", e)


def run_factory():
    try:
        mod = import_module("corerec.engines.contentFilterEngine.cr_contentFilterFactory")
        CFF = getattr(mod, "ContentFilterFactory")
    except Exception as e:
        print("ContentFilterFactory not available:", e)
        return
    try:
        # Minimal config that may be supported by factory when wired
        inst = CFF.create_recommender("tfidf", params={})
        print("Factory created:", type(inst).__name__)
    except Exception as e:
        print("Factory create skipped:", e)


if __name__ == "__main__":
    run_tfidf()
    run_factory() 