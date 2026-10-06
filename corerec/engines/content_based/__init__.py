"""
Content-Based Filtering Engine
==============================

Recommends items by the similarity of their text, which works for items no
one has interacted with yet (cold start).

Usage:
------
    from corerec.engines.content_based import TFIDFRecommender

    model = TFIDFRecommender()
    model.fit(items=[101, 102], docs={101: "action film", 102: "romantic comedy"})
    model.recommend_by_text(query_text="action", top_n=5)

Author: Vishesh Yadav
"""

from .tfidf_recommender import TFIDFRecommender

__all__ = ["TFIDFRecommender"]
