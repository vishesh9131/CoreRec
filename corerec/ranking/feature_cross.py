"""
Feature Cross Ranking

Ranking with explicit feature interactions (DCN-style).
Useful when you have rich user/item features and want to
capture their interactions.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from corerec.retrieval.base import Candidate, RetrievalResult
from .base import BaseRanker, RankedCandidate, RankingResult


class FeatureCrossRanker(BaseRanker):
    """
    Ranker that explicitly models feature interactions.
    
    Inspired by DCN (Deep & Cross Network), this ranker computes
    scores by considering both individual features and their
    cross-products. Good for capturing "users who buy X also like Y".
    
    Can be used with:
    - Pre-defined feature crosses (manual feature engineering)
    - Learned crosses (if you have a trained DCN model)
    - Simple polynomial features
    
    Example::

        ranker = FeatureCrossRanker(
            feature_extractor=extract_user_item_features,
            crosses=[
                ('user_age_bucket', 'item_category'),
                ('user_gender', 'item_brand'),
            ],
            weights={'user_age_bucket:item_category': 0.3, ...}
        )
        ranker.fit()
        ranked = ranker.rank(candidates, user_context)
    """
    
    def __init__(
        self,
        feature_extractor: Optional[Callable[[Any, Dict], Dict]] = None,
        crosses: Optional[List[tuple]] = None,
        weights: Optional[Dict[str, float]] = None,
        model: Optional[Any] = None,
        name: str = "feature_cross",
    ):
        """
        Args:
            feature_extractor: function(item_id, context) -> feature_dict
            crosses: list of feature name tuples to cross
            weights: weights for each feature/cross (learned or manual)
            model: trained model with predict() (overrides manual weights)
            name: identifier for this ranker
        """
        super().__init__(name=name)
        
        self.feature_extractor = feature_extractor
        self.crosses = crosses or []
        self.weights = weights or {}
        self.model = model
    
    def fit(
        self,
        feature_extractor: Optional[Callable] = None,
        crosses: Optional[List[tuple]] = None,
        weights: Optional[Dict[str, float]] = None,
        model: Optional[Any] = None,
        **kwargs
    ) -> "FeatureCrossRanker":
        """
        Configure the ranker.
        
        Args:
            feature_extractor: feature extraction function
            crosses: feature crosses to compute
            weights: feature weights
            model: trained model
        """
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        if crosses is not None:
            self.crosses = crosses
        if weights is not None:
            self.weights = weights
        if model is not None:
            self.model = model
        
        self._is_fitted = True
        return self
    
    def rank(
        self,
        candidates: Union[List[Candidate], RetrievalResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rank candidates using feature crosses.
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        candidates = self._candidates_to_list(candidates)
        context = context or {}
        
        scored = []
        for c in candidates:
            # extract base features
            if self.feature_extractor is not None:
                features = self.feature_extractor(c.item_id, context)
            else:
                features = {}
            
            features['retrieval_score'] = c.score
            
            # compute crossed features
            crossed = self._compute_crosses(features)
            features.update(crossed)
            
            # score
            if self.model is not None:
                score = self._score_with_model(features)
            else:
                score = self._score_with_weights(features)
            
            scored.append(RankedCandidate(
                item_id=c.item_id,
                score=float(score),
                retrieval_score=c.score,
                features=features,
            ))
        
        # sort by score descending
        scored.sort(reverse=True)
        
        # assign ranks
        for i, rc in enumerate(scored):
            rc.rank = i + 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RankingResult(
            candidates=scored,
            query_id=context.get('user_id'),
            ranker_name=self.name,
            timing_ms=elapsed,
        )
    
    def _compute_crosses(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Compute feature crosses."""
        crossed = {}
        
        for cross in self.crosses:
            # cross is a tuple of feature names
            cross_name = ":".join(cross)
            
            # check all features exist
            if all(f in features for f in cross):
                # compute cross value
                values = [features[f] for f in cross]
                
                # for categorical: concatenate
                # for numeric: multiply
                if all(isinstance(v, (int, float)) for v in values):
                    cross_value = np.prod(values)
                else:
                    cross_value = "_".join(str(v) for v in values)
                
                crossed[cross_name] = cross_value
        
        return crossed
    
    def _score_with_weights(self, features: Dict[str, Any]) -> float:
        """Score using manual weights."""
        score = 0.0
        
        for name, value in features.items():
            weight = self.weights.get(name, 0.0)
            
            if isinstance(value, (int, float)):
                score += weight * value
            elif isinstance(value, str):
                # categorical: check if specific value has weight
                specific_key = f"{name}={value}"
                score += self.weights.get(specific_key, 0.0)
        
        return score
    
    def _score_with_model(self, features: Dict[str, Any]) -> float:
        """Score using trained model."""
        if hasattr(self.model, 'predict_proba'):
            X = self._features_to_array(features)
            proba = self.model.predict_proba(X)
            return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        elif hasattr(self.model, 'predict'):
            X = self._features_to_array(features)
            return float(self.model.predict(X)[0])
        elif callable(self.model):
            return float(self.model(features))
        else:
            raise ValueError("Model must have predict() or be callable")
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features to array for sklearn-style models."""
        # extract numeric features
        values = []
        for k, v in sorted(features.items()):
            if isinstance(v, (int, float)):
                values.append(v)
        return np.array([values])
