"""
Batch Inference Engine

Optimized batch inference for high-throughput serving.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import List, Dict, Any, Tuple
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class BatchInferenceEngine:
    """
    Optimized batch inference engine.
    
    Provides efficient batch processing for high-throughput scenarios.
    
    Example:
        from corerec.serving import BatchInferenceEngine
        
        engine = BatchInferenceEngine(model, batch_size=1024)
        
        # Batch predictions
        pairs = [(1,10), (1,11), (2,10), (2,11)]
        scores = engine.batch_predict(pairs)
        
        # Batch recommendations
        user_ids = [1, 2, 3, 4, 5]
        recs = engine.batch_recommend(user_ids, top_k=10)
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(self, model, batch_size: int = 256, num_workers: int = 4):
        """
        Initialize batch inference engine.
        
        Args:
            model: Trained recommendation model
            batch_size: Batch size for inference
            num_workers: Number of parallel workers
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set model to eval mode if it's a PyTorch model
        if hasattr(model, 'eval'):
            model.eval()
    
    def batch_predict(self, pairs: List[Tuple[Any, Any]]) -> List[float]:
        """
        Batch predictions for multiple user-item pairs.
        
        Args:
            pairs: List of (user_id, item_id) tuples
            
        Returns:
            List of predicted scores
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # If model has native batch_predict, use it
        if hasattr(self.model, 'batch_predict'):
            return self.model.batch_predict(pairs)
        
        # Otherwise, batch manually
        scores = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            
            # Parallel processing within batch
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_scores = list(executor.map(
                    lambda p: self.model.predict(p[0], p[1]),
                    batch
                ))
            
            scores.extend(batch_scores)
        
        return scores
    
    def batch_recommend(self, user_ids: List[Any], top_k: int = 10) -> Dict[Any, List]:
        """
        Batch recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            top_k: Number of recommendations per user
            
        Returns:
            Dictionary mapping user_id to recommendations
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # If model has native batch_recommend, use it
        if hasattr(self.model, 'batch_recommend'):
            return self.model.batch_recommend(user_ids, top_k)
        
        # Otherwise, use parallel processing
        recommendations = {}
        
        def get_recommendations(user_id):
            return (user_id, self.model.recommend(user_id, top_k))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = executor.map(get_recommendations, user_ids)
            recommendations = dict(results)
        
        return recommendations
    
    def batch_predict_parallel(self, pairs: List[Tuple[Any, Any]], 
                               num_processes: int = 4) -> List[float]:
        """
        Parallel batch predictions using multiprocessing.
        
        Useful for CPU-bound models.
        
        Args:
            pairs: List of (user_id, item_id) tuples
            num_processes: Number of processes
            
        Returns:
            List of predicted scores
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        from multiprocessing import Pool
        
        with Pool(processes=num_processes) as pool:
            scores = pool.starmap(self.model.predict, pairs)
        
        return scores

