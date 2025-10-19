"""
Unified Base Recommender Interface

This provides a standardized API that all recommendation models must implement,
ensuring consistency across different algorithm families.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import pandas as pd
from pathlib import Path
import pickle
import json


class BaseRecommender(ABC):
    """
    Unified base class for ALL recommendation models in CoreRec.
    
    This enforces consistent API across:
    - Collaborative filtering models
    - Content-based models
    - Hybrid models
    - Deep learning models
    - Graph-based models
    
    Architecture:
    
    ┌─────────────────────────────────────┐
    │      BaseRecommender                │
    │  (Unified Interface)                │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────▼─────┐      ┌──────▼────┐
    │ PyTorch  │      │ Traditional│
    │ Models   │      │   Models   │
    └────┬─────┘      └──────┬────┘
         │                   │
    ┌────▼────┐      ┌──────▼────┐
    │NCF,DLRM │      │  SVD, ALS │
    │DeepFM...│      │  KNN...   │
    └─────────┘      └───────────┘
    
    Standard Methods:
        - fit(): Train the model
        - predict(): Score user-item pairs
        - recommend(): Generate top-K recommendations
        - save(): Persist model to disk
        - load(): Load model from disk
        - batch_predict(): Efficient batch scoring
        - batch_recommend(): Efficient batch recommendations
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, name: str = "BaseRecommender", verbose: bool = False):
        """
        Initialize base recommender.
        
        Args:
            name: Model name for identification
            verbose: Whether to print training logs
        """
        self.name = name
        self.verbose = verbose
        self.is_fitted = False
        self._version = "1.0.0"
    
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, Dict, Any], **kwargs) -> 'BaseRecommender':
        """
        Train the recommendation model.
        
        Args:
            data: Training data (DataFrame, dict, or custom format)
            **kwargs: Additional training parameters
            
        Returns:
            self: For method chaining
            
        Example:
            model.fit(train_data, epochs=10).save('model.pkl')
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """
        Predict score for a single user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted score (higher = more relevant)
            
        Example:
            score = model.predict(user_id=123, item_id=456)
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: Any, top_k: int = 10, 
                  exclude_items: Optional[List[Any]] = None,
                  **kwargs) -> List[Any]:
        """
        Generate top-K item recommendations for a user.
        
        Args:
            user_id: User identifier
            top_k: Number of recommendations to generate
            exclude_items: Items to exclude from recommendations
            **kwargs: Additional recommendation parameters
            
        Returns:
            List of recommended item IDs (sorted by relevance)
            
        Example:
            recs = model.recommend(user_id=123, top_k=10)
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path], format: str = 'pickle') -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
            format: Save format ('pickle', 'json', 'torch')
            
        Example:
            model.save('models/ncf_model.pkl')
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'BaseRecommender':
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded model instance
            
        Example:
            model = NCF.load('models/ncf_model.pkl')
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass
    
    # Non-abstract convenience methods
    
    def batch_predict(self, pairs: List[Tuple[Any, Any]], **kwargs) -> List[float]:
        """
        Predict scores for multiple user-item pairs efficiently.
        
        Args:
            pairs: List of (user_id, item_id) tuples
            **kwargs: Additional parameters
            
        Returns:
            List of predicted scores
            
        Example:
            scores = model.batch_predict([(1,10), (1,11), (2,10)])
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return [self.predict(user_id, item_id, **kwargs) for user_id, item_id in pairs]
    
    def batch_recommend(self, user_ids: List[Any], top_k: int = 10, **kwargs) -> Dict[Any, List[Any]]:
        """
        Generate recommendations for multiple users efficiently.
        
        Args:
            user_ids: List of user identifiers
            top_k: Number of recommendations per user
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping user_id to list of recommended items
            
        Example:
            recs = model.batch_recommend([1, 2, 3], top_k=5)
            # {1: [10,11,12,13,14], 2: [20,21,22,23,24], ...}
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return {uid: self.recommend(uid, top_k, **kwargs) for uid in user_ids}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and information.
        
        Returns:
            Dictionary containing model info
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return {
            'name': self.name,
            'version': self._version,
            'is_fitted': self.is_fitted,
            'model_type': self.__class__.__name__,
            'module': self.__class__.__module__
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"

