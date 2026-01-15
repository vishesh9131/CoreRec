# moe_recommender.py

from corerec.engines.collaborative.base_recommender import BaseRecommender
from typing import List, Dict
from scipy.sparse import csr_matrix
import logging
from typing import List, Optional


class MixtureOfExpertsRecommender(BaseRecommender):
    def __init__(self, experts: List[BaseRecommender]):
        """
        Initialize the Mixture of Experts Recommender.

        Args:
            experts (List[BaseRecommender]): List of expert recommenders.
        """
        self.experts = experts
        self.weights = [1.0 / len(experts) for _ in experts]  # Equal initial weighting

    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train all expert recommenders.

        Args:
            interaction_matrix (csr_matrix): User-item interaction matrix.
            user_ids (List[int]): List of user IDs.
            item_ids (List[int]): List of item IDs.
        """
        for expert in self.experts:
            expert.fit(interaction_matrix, user_ids, item_ids)

    def recommend(
        self, user_id: int, top_n: int = 10, exclude_items: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate recommendations by aggregating experts' suggestions.

        Args:
            user_id (int): The ID of the user.
            top_n (int): The number of recommendations to generate.
            exclude_items (Optional[List[int]]): List of item IDs to exclude.

        Returns:
            List[int]: List of recommended item IDs.
        """
        combined_scores: Dict[int, float] = {}
        for weight, expert in zip(self.weights, self.experts):
            recs = expert.recommend(user_id, top_n=top_n * 2, exclude_items=exclude_items)
            for item in recs:
                combined_scores[item] = combined_scores.get(item, 0) + weight

        # Sort items based on combined scores
        ranked_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item for item, score in ranked_items[:top_n]]
        return recommendations
