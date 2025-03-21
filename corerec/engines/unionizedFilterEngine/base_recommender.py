from abc import ABC, abstractmethod
from typing import List, Optional
from scipy.sparse import csr_matrix
from .device_manager import DeviceManager

class BaseRecommender(ABC):
    def __init__(self, device: str = 'auto'):
        """
        Initialize the base recommender.
        
        Args:
            device: Computation device to use ('cpu', 'cuda', 'mps', etc.)
        """
        self.device_manager = DeviceManager(preferred_device=device)
        self.device = self.device_manager.active_device
    
    def to_device(self, data):
        """Move data to the active device."""
        return self.device_manager.to_device(data)
    
    def create_tensor(self, data, dtype=None):
        """Create a tensor on the active device."""
        return self.device_manager.create_tensor(data, dtype=dtype)
    
    @abstractmethod
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the recommender system using the provided interaction matrix.
        
        Parameters:
        - interaction_matrix (csr_matrix): Sparse matrix of user-item interactions.
        - user_ids (List[int]): List of user IDs.
        - item_ids (List[int]): List of item IDs.
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Generate recommendations for a user.
        
        Parameters:
        - user_id (int): The ID of the user.
        - top_n (int): The number of recommendations to generate.
        
        Returns:
        - List[int]: List of recommended item IDs.
        """
        pass 