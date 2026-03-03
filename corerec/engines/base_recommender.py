"""Re-export BaseRecommender so relative imports within engines/ work."""
from corerec.api.base_recommender import BaseRecommender

__all__ = ["BaseRecommender"]
