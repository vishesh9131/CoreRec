from typing import Dict, Any
from corerec.base_recommender import BaseCorerec
from .tfidf_recommender import TFIDFRecommender
# Import other content-based recommender classes here as they are implemented

class ContentFilterFactory:
    @staticmethod
    def get_recommender(config: Dict[str, Any]) -> BaseCorerec:
        method = config.get("method")
        params = config.get("params", {})

        if method == "tfidf":
            return TFIDFRecommender(
                feature_matrix=params.get("feature_matrix")
            )
        # Add more elif blocks for additional content-based methods
        else:
            raise ValueError(f"Unsupported content-based filtering method: {method}")