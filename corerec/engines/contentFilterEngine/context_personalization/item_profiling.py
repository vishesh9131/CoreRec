# item_profiling implementation
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ItemProfilingRecommender:
    """
    A recommender system that builds and maintains detailed item profiles for context-aware recommendations.

    This recommender system creates comprehensive profiles for items by analyzing their features,
    usage patterns, and performance across different contexts. It uses these profiles to make
    more accurate recommendations based on contextual similarities.

    Attributes:
        item_profiles (Dict[int, Dict]): Detailed profiles for each item containing:
            - Static features (inherent item characteristics)
            - Dynamic features (usage patterns, ratings)
            - Contextual performance metrics
        context_performance (Dict[int, Dict]): Performance metrics for items in different contexts
        feature_importance (Dict[str, float]): Learned importance of different features

    Methods:
        build_profile: Creates or updates an item's profile
        update_context_performance: Updates item performance metrics for specific contexts
        get_context_similarity: Calculates similarity between contexts
        recommend: Generates recommendations using item profiles and current context

    Example:
        >>> profiler = ItemProfilingRecommender()
        >>> profiler.build_profile(item_id=123, features={...}, context_data={...})
        >>> recommendations = profiler.recommend(user_id=456, context={"time": "evening"})
    """

    def __init__(self):
        """
        Initialize the item profiling recommender.
        """
        self.item_profiles: Dict[int, Dict[str, Any]] = {}

    def fit(self, data: Dict[int, List[int]], item_features: Dict[int, Dict[str, Any]]):
        """
        Train the recommender system by building item profiles.

        Parameters:
        - data (dict): The data used for training the model, containing user interactions.
        - item_features (dict): A dictionary mapping item IDs to their features.
        """
        if not data:
            raise ValueError("Training data cannot be empty.")
        if not item_features:
            raise ValueError("Item features cannot be empty.")

        # Example: Count-based item profiling
        for user_id, items in data.items():
            for item_id in items:
                if item_id not in self.item_profiles:
                    self.item_profiles[item_id] = {}
                for feature, value in item_features.get(item_id, {}).items():
                    self.item_profiles[item_id][feature] = (
                        self.item_profiles[item_id].get(feature, 0) + 1
                    )

    def recommend(self, item_id: int, top_n: int = 10) -> List[int]:
        """
        Recommend items similar to the given item based on item profiles.

        Parameters:
        - item_id (int): The ID of the item to find similar items for.
        - top_n (int): Number of top recommendations to return.

        Returns:
        - List[int]: List of recommended item IDs.
        """
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        
        if not self.item_profiles:
            logger.warning("No item profiles found. Please fit the model first.")
            return []
        
        if item_id not in self.item_profiles:
            logger.warning(f"Item {item_id} not found in profiles.")
            return []
        
        logger.info(f"Generating recommendations for item {item_id} based on item profiles.")
        
        # Get the target item's profile
        target_profile = self.item_profiles[item_id]
        
        # Calculate similarity with other items
        similarities = {}
        for other_item_id, other_profile in self.item_profiles.items():
            if other_item_id == item_id:
                continue
            
            # Simple similarity based on shared features
            similarity = 0.0
            common_features = set(target_profile.keys()) & set(other_profile.keys())
            if common_features:
                for feature in common_features:
                    similarity += min(target_profile[feature], other_profile[feature])
            
            similarities[other_item_id] = similarity
        
        # Sort by similarity and return top_n
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in sorted_items[:top_n]]
        
        logger.info(f"Generated {len(recommendations)} recommendations.")
        return recommendations
