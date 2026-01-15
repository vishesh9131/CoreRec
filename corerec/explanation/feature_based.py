"""
Feature-based Explanations

Generate explanations based on matching features between
user preferences and item attributes.
"""

from typing import Any, Callable, Dict, List, Optional

from .base import BaseExplainer, Explanation


class FeatureExplainer(BaseExplainer):
    """
    Generates explanations based on feature matches.
    
    Looks at which features of the item match user preferences
    and generates explanations like:
    - "Because you liked action movies"
    - "Similar to items in your history"
    - "Matches your preferred price range"
    
    Example:
        explainer = FeatureExplainer(
            item_features=item_feature_dict,
            user_preferences=user_pref_dict,
            templates={
                'category': "Because you like {value}",
                'brand': "From {value}, a brand you've purchased before",
            }
        )
        
        explanation = explainer.explain(item_id=123, context={'user_id': 456})
    """
    
    def __init__(
        self,
        item_features: Optional[Dict[Any, Dict[str, Any]]] = None,
        user_preferences: Optional[Dict[Any, Dict[str, Any]]] = None,
        templates: Optional[Dict[str, str]] = None,
        feature_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
        preference_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
        name: str = "feature_explainer",
    ):
        """
        Args:
            item_features: dict mapping item_id -> feature_dict
            user_preferences: dict mapping user_id -> preference_dict
            templates: templates for each feature type
            feature_extractor: function(item_id) -> features (alternative to dict)
            preference_extractor: function(user_id) -> preferences
            name: identifier
        """
        super().__init__(name=name)
        
        self.item_features = item_features or {}
        self.user_preferences = user_preferences or {}
        self.feature_extractor = feature_extractor
        self.preference_extractor = preference_extractor
        
        self.templates = templates or {
            'category': "Because you like {value}",
            'genre': "Matches your interest in {value}",
            'brand': "From {value}, a brand you trust",
            'price_range': "Within your preferred budget",
            'default': "Based on your preferences",
        }
    
    def explain(
        self,
        item_id: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> Explanation:
        """Generate feature-based explanation."""
        user_id = context.get('user_id')
        
        # get item features
        if self.feature_extractor:
            item_feats = self.feature_extractor(item_id)
        else:
            item_feats = self.item_features.get(item_id, {})
        
        # get user preferences
        if self.preference_extractor and user_id:
            user_prefs = self.preference_extractor(user_id)
        elif user_id:
            user_prefs = self.user_preferences.get(user_id, {})
        else:
            user_prefs = {}
        
        # find matching features
        matches = []
        for feat_name, feat_value in item_feats.items():
            if feat_name in user_prefs:
                user_pref = user_prefs[feat_name]
                
                # check if they match (simple equality or containment)
                if self._features_match(feat_value, user_pref):
                    matches.append((feat_name, feat_value))
        
        # generate explanation text
        if matches:
            # use the most specific match
            feat_name, feat_value = matches[0]
            template = self.templates.get(feat_name, self.templates['default'])
            text = template.format(value=feat_value, feature=feat_name)
            exp_type = f"feature_{feat_name}"
        else:
            text = "Recommended for you"
            exp_type = "generic"
        
        return Explanation(
            item_id=item_id,
            text=text,
            explanation_type=exp_type,
            features=item_feats,
        )
    
    def _features_match(self, item_value: Any, user_pref: Any) -> bool:
        """Check if item feature matches user preference."""
        # exact match
        if item_value == user_pref:
            return True
        
        # item value in user preference list
        if isinstance(user_pref, (list, set)):
            return item_value in user_pref
        
        # user preference in item value (e.g., category hierarchy)
        if isinstance(item_value, str) and isinstance(user_pref, str):
            return user_pref.lower() in item_value.lower()
        
        return False


class HistoryExplainer(BaseExplainer):
    """
    Explains based on similar items in user history.
    
    Generates explanations like:
    - "Because you watched Movie X"
    - "Similar to items you purchased recently"
    """
    
    def __init__(
        self,
        user_history: Optional[Dict[Any, List[Any]]] = None,
        item_similarity: Optional[Callable[[Any, Any], float]] = None,
        item_names: Optional[Dict[Any, str]] = None,
        name: str = "history_explainer",
    ):
        """
        Args:
            user_history: dict mapping user_id -> list of interacted item_ids
            item_similarity: function(item_a, item_b) -> similarity score
            item_names: dict mapping item_id -> display name
            name: identifier
        """
        super().__init__(name=name)
        
        self.user_history = user_history or {}
        self.item_similarity = item_similarity
        self.item_names = item_names or {}
    
    def explain(
        self,
        item_id: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> Explanation:
        """Generate history-based explanation."""
        user_id = context.get('user_id')
        
        if not user_id or user_id not in self.user_history:
            return Explanation(
                item_id=item_id,
                text="Recommended for you",
                explanation_type="generic",
            )
        
        history = self.user_history[user_id]
        
        if not history:
            return Explanation(
                item_id=item_id,
                text="Recommended for you",
                explanation_type="generic",
            )
        
        # find most similar item in history
        best_match = None
        best_sim = 0.0
        
        if self.item_similarity:
            for hist_item in history[-10:]:  # look at recent history
                sim = self.item_similarity(item_id, hist_item)
                if sim > best_sim:
                    best_sim = sim
                    best_match = hist_item
        else:
            # no similarity function, just use most recent
            best_match = history[-1]
        
        if best_match:
            match_name = self.item_names.get(best_match, str(best_match))
            text = f"Because you liked {match_name}"
            supporting = [best_match]
        else:
            text = "Based on your history"
            supporting = []
        
        return Explanation(
            item_id=item_id,
            text=text,
            explanation_type="history",
            confidence=best_sim if self.item_similarity else 0.5,
            supporting_items=supporting,
        )
