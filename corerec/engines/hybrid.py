# engines/hybrid.py
from typing import List, Optional, Any
import numpy as np
import logging

from scipy.sparse import csr_matrix

from corerec.engines.unionizedFilterEngine.base_recommender import BaseRecommender
from corerec.engines.contentFilterEngine.base_recommender import BaseRecommender as ContentBaseRecommender

class HybridEngine:
    """
    Hybrid Recommendation Engine that combines Collaborative Filtering and Content-Based Filtering.
    """

    def __init__(
        self, 
        collaborative_engine: BaseRecommender, 
        content_engine: Optional[ContentBaseRecommender] = None, 
        alpha: float = 0.5
    ):
        """
        Initializes the HybridEngine with specified collaborative and content-based engines.
        
        Parameters:
        - collaborative_engine (BaseRecommender): Collaborative filtering component.
        - content_engine (Optional[ContentBaseRecommender]): Content-based filtering component.
        - alpha (float): Weighting factor for combining recommendations (0 <= alpha <= 1).
                           alpha=1 uses only collaborative filtering,
                           alpha=0 uses only content-based filtering.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha should be between 0 and 1.")
        
        self.collaborative_engine = collaborative_engine
        self.content_engine = content_engine
        self.alpha = alpha
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"HybridEngine initialized with alpha={self.alpha}.")

    def train(
        self, 
        interaction_matrix: Any, 
        user_ids: List[int], 
        item_ids: List[int]
    ):
        """
        Trains both collaborative and content-based engines.
        
        Parameters:
        - interaction_matrix (sparse matrix): User-item interaction matrix.
        - user_ids (List[int]): List of user IDs corresponding to the rows of interaction_matrix.
        - item_ids (List[int]): List of item IDs corresponding to the columns of interaction_matrix.
        """
        if not isinstance(interaction_matrix, csr_matrix):
            self.logger.warning("interaction_matrix is not a CSR matrix. Converting to CSR format.")
            try:
                interaction_matrix = csr_matrix(interaction_matrix)
                self.logger.info("Successfully converted interaction_matrix to CSR format.")
            except Exception as e:
                self.logger.error(f"Failed to convert interaction_matrix to CSR format: {e}")
                raise ValueError("Failed to convert interaction_matrix to CSR sparse matrix.") from e

        # Train Collaborative Engine
        self.collaborative_engine.fit(interaction_matrix, user_ids, item_ids)
        self.logger.info("Collaborative engine training completed.")

        # Train Content-Based Engine if it exists
        if self.content_engine is not None:
            self.content_engine.fit(item_ids)
            self.logger.info("Content-based engine training completed.")

    def recommend(
        self, 
        user_id: int, 
        top_n: int = 10, 
        exclude_items: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generates hybrid recommendations for a given user.
        
        Parameters:
        - user_id (int): The ID of the user.
        - top_n (int): The number of recommendations to generate.
        - exclude_items (Optional[List[int]]): List of item IDs to exclude from recommendations.
        
        Returns:
        - List[int]: Combined list of recommended item IDs.
        """
        if exclude_items is None:
            exclude_items = []

        # Collaborative Filtering Recommendations
        collab_recs = self.collaborative_engine.recommend(user_id, top_n * 2)
        self.logger.debug(f"Collaborative recommendations: {collab_recs}")

        # Content-Based Filtering Recommendations
        try:
            # Assuming user has a favorite item(s). For simplicity, taking first interaction.
            # This should be replaced with actual user profile interactions.
            favorite_item_id = collab_recs[0] if collab_recs else 0  # Default to 0 if no collab recs
            content_recs = self.content_engine.recommend([favorite_item_id], top_n * 2)
            self.logger.debug(f"Content-Based recommendations: {content_recs}")
        except Exception as e:
            self.logger.error(f"Content-Based Filtering recommendation failed: {e}")
            content_recs = []
        
        # Combine Recommendations with Weighting
        hybrid_recs = self._combine_recommendations(collab_recs, content_recs)
        self.logger.debug(f"Combined hybrid recommendations: {hybrid_recs}")
        
        # Exclude Already Interacted Items
        final_recs = [item for item in hybrid_recs if item not in exclude_items]
        final_recs = final_recs[:top_n]
        
        self.logger.info(f"Final Hybrid Recommendations: {final_recs}")
        return final_recs

    def _combine_recommendations(
        self, 
        collab_recs: List[int], 
        content_recs: List[int]
    ) -> List[int]:
        """
        Combines collaborative and content-based recommendations based on the weighting factor alpha.
        
        Parameters:
        - collab_recs (List[int]): Recommendations from collaborative filtering.
        - content_recs (List[int]): Recommendations from content-based filtering.
        
        Returns:
        - List[int]: Combined list of recommended item IDs.
        """
        self.logger.debug("Combining recommendations from both engines.")
        
        # Convert lists to sets for weighted combination
        collab_set = set(collab_recs)
        content_set = set(content_recs)
        
        # Weighted Combination
        combined_scores = {}
        
        for item in collab_set.union(content_set):
            score = 0
            if item in collab_set:
                score += self.alpha
            if item in content_set:
                score += (1 - self.alpha)
            combined_scores[item] = score
        
        # Sort items based on combined scores
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        combined_recs = [item for item, score in sorted_items]
        
        return combined_recs