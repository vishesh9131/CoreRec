"""
Pointwise Ranking

Scores each candidate independently. Simple but effective,
and easy to train with standard classification/regression losses.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from corerec.retrieval.base import Candidate, RetrievalResult
from .base import BaseRanker, RankedCandidate, RankingResult


class PointwiseRanker(BaseRanker):
    """
    Ranks candidates by scoring each one independently.
    
    The scoring function can be:
    - A trained ML model (sklearn, torch, etc)
    - A simple callable that takes features and returns score
    - A weighted combination of feature values
    
    Example::

        # with a trained model
        ranker = PointwiseRanker(model=trained_model)
        ranker.fit(feature_extractor=my_feature_fn)
        
        # with a simple scoring function
        ranker = PointwiseRanker(
            score_fn=lambda feats: feats['quality'] * 0.7 + feats['recency'] * 0.3
        )
        ranker.fit()
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        score_fn: Optional[Callable[[Dict], float]] = None,
        feature_extractor: Optional[Callable[[Any, Dict], Dict]] = None,
        name: str = "pointwise",
    ):
        """
        Args:
            model: ML model with predict() or predict_proba() method
            score_fn: custom scoring function (features -> score)
            feature_extractor: function to extract features from (item_id, context)
            name: identifier for this ranker
        
        Provide either model or score_fn.
        """
        super().__init__(name=name)
        
        self.model = model
        self.score_fn = score_fn
        self.feature_extractor = feature_extractor
        
        if model is None and score_fn is None:
            # default: use retrieval score
            self.score_fn = lambda feats: feats.get('retrieval_score', 0.0)
    
    def fit(
        self,
        model: Optional[Any] = None,
        feature_extractor: Optional[Callable] = None,
        **kwargs
    ) -> "PointwiseRanker":
        """
        Configure the ranker.
        
        Args:
            model: ML model to use for scoring
            feature_extractor: function(item_id, context) -> feature_dict
        """
        if model is not None:
            self.model = model
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        
        self._is_fitted = True
        return self
    
    def rank(
        self,
        candidates: Union[List[Candidate], RetrievalResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Score and rank candidates.
        
        Args:
            candidates: candidates from retrieval
            context: user features, session info, etc.
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        candidates = self._candidates_to_list(candidates)
        context = context or {}
        
        scored = []
        for c in candidates:
            # extract features
            if self.feature_extractor is not None:
                features = self.feature_extractor(c.item_id, context)
            else:
                features = {'retrieval_score': c.score}
            
            # add retrieval score to features
            features['retrieval_score'] = c.score
            
            # compute score
            if self.model is not None:
                score = self._score_with_model(features)
            else:
                score = self.score_fn(features)
            
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
    
    def _score_with_model(self, features: Dict[str, Any]) -> float:
        """Score using the ML model."""
        # convert features to format model expects
        if hasattr(self.model, 'predict_proba'):
            # classifier - use probability
            X = self._features_to_array(features)
            proba = self.model.predict_proba(X)
            return float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        elif hasattr(self.model, 'predict'):
            X = self._features_to_array(features)
            pred = self.model.predict(X)
            return float(pred[0])
        elif callable(self.model):
            return float(self.model(features))
        else:
            raise ValueError("Model must have predict() or be callable")
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dict to array for sklearn-style models."""
        # simple: just use numeric values in order
        values = [v for v in features.values() if isinstance(v, (int, float))]
        return np.array([values])
    
    def rank_batch(
        self,
        candidate_batches: List[Union[List[Candidate], RetrievalResult]],
        contexts: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[RankingResult]:
        """
        Batch ranking - can be more efficient with batched feature extraction.
        """
        self._check_fitted()
        
        if contexts is None:
            contexts = [{}] * len(candidate_batches)
        
        # for now, just loop - subclasses can optimize
        return [
            self.rank(candidates, context, **kwargs)
            for candidates, context in zip(candidate_batches, contexts)
        ]
