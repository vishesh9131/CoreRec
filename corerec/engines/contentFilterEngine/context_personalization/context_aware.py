import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import os

class ContextAwareRecommender:
    """
    A context-aware recommender system that adapts recommendations based on contextual factors.
    
    This recommender system incorporates contextual information to provide more relevant 
    recommendations by adjusting feature weights based on the current context. It considers
    various contextual factors such as time, location, user state, and other environmental 
    variables to modify the importance of different item features.
    
    Attributes:
        context_factors (Dict[str, Dict]): Mapping of context factors to their value-specific weights.
            Format: {
                "factor_name": {
                    "value": {
                        "feature": weight
                    }
                }
            }
        item_features (Dict[int, Dict]): Mapping of item IDs to their feature dictionaries.
            Format: {
                item_id: {
                    "feature_name": value
                }
            }
        feature_weights (Dict[str, float]): Current active feature weights based on context.
    
    Methods:
        update_context: Updates the current context and recalculates feature weights.
        recommend: Generates recommendations considering the current context.
        _initialize_feature_weights: Initializes weights based on context.
        _encode_item_features: Encodes and weights item features.
    """
    
    def __init__(
        self, 
        context_config_path: str, 
        item_features: Dict[int, Dict[str, Any]]
    ):
        """
        Initialize the context-aware recommender with a configuration file for context factors and item features.

        Parameters:
        - context_config_path (str): Path to the JSON configuration file for context factors and weights.
        - item_features (dict): A dictionary of item features.
        """
        self.context_factors = self._load_context_config(context_config_path)
        self.item_features = item_features
        self.user_profiles = defaultdict(lambda: defaultdict(float))
        self.feature_weights = self._initialize_feature_weights()

    def _load_context_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load context factors and their configurations from a JSON file.

        Parameters:
        - config_path (str): Path to the JSON configuration file.

        Returns:
        - Dict[str, Any]: Configuration for context factors.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            return {}
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    def _initialize_feature_weights(self, current_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Initialize and calculate feature weights based on the current context.
        
        This method processes the current context to determine appropriate weights for different
        item features. It combines weights from multiple context factors when they affect the
        same feature.
        
        Args:
            current_context (Optional[Dict[str, Any]]): Dictionary containing current context
                information. Keys are context factor names, values are the current factor values.
                If None, uses default context.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to their calculated weights
                based on the current context.
        
        Example:
            >>> context = {"time": "evening", "location": "home"}
            >>> recommender._initialize_feature_weights(context)
            {"genre_action": 1.2, "genre_drama": 0.8, "length": 1.5}
        """
        weights = {}
        context = current_context or self.context_factors.get("default", {})
        for factor, value in context.items():
            factor_config = self.context_factors.get(factor, {})
            value_weights = factor_config.get(str(value), {})
            for feature, weight in value_weights.items():
                weights[feature] = weight
        return weights

    def _encode_item_features(self, item_id: int) -> Dict[str, float]:
        """
        Encode item features into a weighted feature vector based on current context.
        
        This method transforms an item's raw features into a weighted feature representation,
        applying the current context-dependent weights. It handles both categorical and
        numerical features appropriately.
        
        Args:
            item_id (int): The unique identifier of the item to encode.
        
        Returns:
            Dict[str, float]: Dictionary containing the encoded and weighted features.
                For categorical features: {feature_name_value: weight}
                For numerical features: {feature_name: value * weight}
        
        Example:
            >>> recommender._encode_item_features(123)
            {"genre_action": 1.2, "duration": 90.5, "rating": 4.5}
        
        Note:
            - Categorical features are encoded as separate binary features
            - Numerical features are scaled by their corresponding weights
        """
        features = self.item_features.get(item_id, {})
        encoded = {}
        for feature, value in features.items():
            key = f"{feature}_{value}" if isinstance(value, str) else feature
            weight = self.feature_weights.get(key, 1.0)
            if isinstance(value, str):
                encoded[key] = weight
            else:
                encoded[feature] = value * weight
        return encoded

    def fit(self, data: Dict[int, List[int]]):
        """
        Train the recommender system by building user profiles based on their interactions.

        Parameters:
        - data (dict): The data used for training the model, containing user interactions.
        """
        for user_id, items in data.items():
            for item_id in items:
                encoded_features = self._encode_item_features(item_id)
                for feature, value in encoded_features.items():
                    self.user_profiles[user_id][feature] += value

    def recommend(
        self, 
        user_id: int, 
        context: Optional[Dict[str, Any]] = None, 
        top_n: int = 10
    ) -> List[int]:
        """
        Generate top-N item recommendations for a given user considering context.

        Parameters:
        - user_id (int): The ID of the user.
        - context (dict, optional): The current context to consider. If provided, updates context factors.
        - top_n (int): The number of recommendations to generate.

        Returns:
        - List[int]: List of recommended item IDs.
        """
        if context:
            self.feature_weights = self._initialize_feature_weights(context)
        else:
            self.feature_weights = self._initialize_feature_weights()

        user_profile = self.user_profiles.get(user_id, {})
        if not user_profile:
            return []

        scores = {}
        interacted_items = set()
        # Collect all items the user has interacted with to exclude them from recommendations
        interacted_items = user_profile.get('interacted_items', set())

        for item_id, features in self.item_features.items():
            if item_id in interacted_items:
                continue
            encoded_features = self._encode_item_features(item_id)
            score = 0.0
            for feature, value in encoded_features.items():
                score += user_profile.get(feature, 0.0) * value
            scores[item_id] = score

        # Sort items based on the computed scores in descending order
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return the top-N item IDs
        return [item_id for item_id, score in ranked_items[:top_n]]
