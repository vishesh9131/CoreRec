from typing import Dict, Any
from .base_recommender import BaseRecommender
from .matrix_factorization import MatrixFactorizationRecommender
from .mf_base.user_based_uf import UserBasedCollaborativeFilteringRecommender
from .svd_recommender import SVDRecommender
# Import additional CF recommender classes as implemented


class UnionizedRecommenderFactory:
    @staticmethod
    def get_recommender(config: Dict[str, Any]) -> BaseRecommender:
        method = config.get("method")
        params = config.get("params", {})

        if method == "matrix_factorization":
            return MatrixFactorizationRecommender(
                num_factors=params.get("num_factors", 20),
                learning_rate=params.get("learning_rate", 0.01),
                regularization=params.get("regularization", 0.02),
                epochs=params.get("epochs", 20),
            )
        elif method == "user_based_cf":
            return UserBasedCollaborativeFilteringRecommender(
                similarity_threshold=params.get("similarity_threshold", 0.5)
            )
        elif method == "svd":
            return SVDRecommender(
                num_factors=params.get("num_factors", 20),
                learning_rate=params.get("learning_rate", 0.01),
                regularization=params.get("regularization", 0.02),
                epochs=params.get("epochs", 20),
            )
        # Add more elif blocks for additional CF methods
        else:
            raise ValueError(f"Unsupported Unionized Filtering method: {method}")