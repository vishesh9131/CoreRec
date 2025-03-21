# corerec/engines/unionizedFilterEngine/matrix_factorization_recommender.py
import numpy as np
from typing import List , Optional
from scipy.sparse import csr_matrix
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_base import MatrixFactorizationBase

class MatrixFactorizationRecommender(MatrixFactorizationBase):
    def __init__(self, num_factors: int = 20, learning_rate: float = 0.01, 
                 reg_user: float = 0.02, reg_item: float = 0.02, 
                 epochs: int = 20, early_stopping_rounds: Optional[int] = None, 
                 n_threads: int = 4):
        super().__init__(num_factors, learning_rate, reg_user, reg_item, epochs, early_stopping_rounds, n_threads)
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_items: Optional[List[int]] = None) -> List[int]:
        """
        Generate top-N item recommendations for a given user based on the trained matrix factorization model.

        Parameters:
        - user_id (int): The ID of the user.
        - top_n (int): The number of recommendations to generate.
        - exclude_items (Optional[List[int]]): List of item IDs to exclude from recommendations.

        Returns:
        - List[int]: List of recommended item IDs.
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("The model has not been trained yet. Please call the 'fit' method first.")
        
        # Compute the scores for all items
        user_vector = self.user_factors[user_id]
        scores = self.item_factors.dot(user_vector) + self.user_bias[user_id] + self.item_bias + self.global_bias
        
        # Exclude items if necessary
        if exclude_items:
            scores[exclude_items] = -np.inf  # Assign a very low score to excluded items
        
        # Get the top N item indices
        top_items = np.argsort(scores)[-top_n:][::-1]
        return top_items.tolist()