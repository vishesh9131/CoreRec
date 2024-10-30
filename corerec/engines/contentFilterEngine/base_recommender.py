from abc import ABC, abstractmethod
from typing import List

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, data):
        """
        Train the recommender system using the provided data.
        
        Parameters:
        - data: The data used for training the model.
        """
        pass

    @abstractmethod
    def recommend(self, item_indices: List[int], top_n: int = 10) -> List[int]:
        """
        Generate top-N item recommendations based on content features.
        
        Parameters:
        - item_indices (List[int]): List of item indices to base recommendations on.
        - top_n (int): The number of recommendations to generate.
        
        Returns:
        - List[int]: List of recommended item indices.
        """
        pass
