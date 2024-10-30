from abc import ABC, abstractmethod
from typing import List
from scipy.sparse import csr_matrix

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the recommender system using the provided interaction matrix.
        
        Parameters:
        - interaction_matrix (csr_matrix): User-item interaction matrix.
        - user_ids (List[int]): List of user IDs.
        - item_ids (List[int]): List of item IDs.
        """
        pass

    @abstractmethod
    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Generate top-N item recommendations for a given user.
        
        Parameters:
        - user_id (int): The ID of the user.
        - top_n (int): The number of recommendations to generate.
        
        Returns:
        - List[int]: List of recommended item IDs.
        """
        pass 