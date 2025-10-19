"""
Evaluation Metrics

Comprehensive metrics for recommendation system evaluation.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import numpy as np
from typing import List, Set, Dict, Any
from collections import defaultdict


class RankingMetrics:
    """
    Ranking metrics for recommendation evaluation.
    
    Implements standard metrics like NDCG, MAP, MRR, Precision, Recall.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    @staticmethod
    def ndcg_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain @ K.
        
        Args:
            predictions: Predicted ranked list of items
            ground_truth: Ground truth relevant items
            k: Cut-off position
            
        Returns:
            NDCG @ K score
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        # Calculate DCG
        dcg = sum([
            (1 if pred in ground_truth_set else 0) / np.log2(i + 2)
            for i, pred in enumerate(predictions)
        ])
        
        # Calculate IDCG
        ideal_length = min(k, len(ground_truth))
        idcg = sum([1 / np.log2(i + 2) for i in range(ideal_length)])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Mean Average Precision @ K.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred in enumerate(predictions):
            if pred in ground_truth_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(ground_truth), k) if ground_truth else 0.0
    
    @staticmethod
    def mrr_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Mean Reciprocal Rank @ K.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        for i, pred in enumerate(predictions):
            if pred in ground_truth_set:
                return 1.0 / (i + 1.0)
        
        return 0.0
    
    @staticmethod
    def precision_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Precision @ K.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        hits = sum(1 for pred in predictions if pred in ground_truth_set)
        return hits / k
    
    @staticmethod
    def recall_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Recall @ K.
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        hits = sum(1 for pred in predictions if pred in ground_truth_set)
        return hits / len(ground_truth) if ground_truth else 0.0
    
    @staticmethod
    def hit_rate_at_k(predictions: List, ground_truth: List, k: int = 10) -> float:
        """
        Hit Rate @ K (binary: 1 if any hit, 0 otherwise).
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        predictions = predictions[:k]
        ground_truth_set = set(ground_truth)
        
        return 1.0 if any(pred in ground_truth_set for pred in predictions) else 0.0


class ClassificationMetrics:
    """
    Classification metrics for binary/multi-class prediction.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    @staticmethod
    def accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate accuracy."""
        return (predictions == ground_truth).mean()
    
    @staticmethod
    def precision(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate precision."""
        true_positives = ((predictions == 1) & (ground_truth == 1)).sum()
        predicted_positives = (predictions == 1).sum()
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    @staticmethod
    def recall(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate recall."""
        true_positives = ((predictions == 1) & (ground_truth == 1)).sum()
        actual_positives = (ground_truth == 1).sum()
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    @staticmethod
    def f1_score(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate F1 score."""
        p = ClassificationMetrics.precision(predictions, ground_truth)
        r = ClassificationMetrics.recall(predictions, ground_truth)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class DiversityMetrics:
    """
    Diversity and novelty metrics for recommendations.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    @staticmethod
    def intra_list_diversity(recommendations: List[List]) -> float:
        """
        Intra-list diversity (average pairwise distance within lists).
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not recommendations:
            return 0.0
        
        diversities = []
        for rec_list in recommendations:
            unique_items = len(set(rec_list))
            total_items = len(rec_list)
            diversities.append(unique_items / total_items if total_items > 0 else 0.0)
        
        return np.mean(diversities)
    
    @staticmethod
    def coverage(all_recommendations: List[List], total_items: int) -> float:
        """
        Catalog coverage (percentage of items ever recommended).
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        recommended_items = set()
        for rec_list in all_recommendations:
            recommended_items.update(rec_list)
        
        return len(recommended_items) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def gini_coefficient(item_counts: Dict[Any, int]) -> float:
        """
        Gini coefficient (measures recommendation concentration).
        
        0 = perfect equality, 1 = perfect inequality
        
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not item_counts:
            return 0.0
        
        counts = sorted(item_counts.values())
        n = len(counts)
        
        cumsum = np.cumsum(counts)
        return (2 * np.sum((np.arange(1, n+1) * counts))) / (n * cumsum[-1]) - (n + 1) / n

