from typing import Dict, Any
from .base_recommender import BaseRecommender
from .tfidf_recommender import TFIDFRecommender
# Import other content-based recommender classes here as they are implemented

class ContentFilterFactory:
    @staticmethod
    def get_recommender(config: Dict[str, Any]) -> BaseRecommender:
        method = config.get("method")
        params = config.get("params", {})

        if method == "tfidf":
            return TFIDFRecommender(
                feature_matrix=params.get("feature_matrix")
            )
        # Add more elif blocks for additional content-based methods
        else:
            raise ValueError(f"Unsupported content-based filtering method: {method}")