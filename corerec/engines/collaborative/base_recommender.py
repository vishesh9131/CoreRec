"""Re-export BaseRecommender so relative imports within collaborative/ work."""
from corerec.api.base_recommender import BaseRecommender

__all__ = ["BaseRecommender"]
