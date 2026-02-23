"""
Pairwise Ranking

Learns preferences between item pairs rather than absolute scores.
Good for learning-to-rank with implicit feedback.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from corerec.retrieval.base import Candidate, RetrievalResult
from .base import BaseRanker, RankedCandidate, RankingResult


class PairwiseRanker(BaseRanker):
    """
    Ranks by comparing pairs of candidates.
    
    Instead of predicting absolute scores, pairwise ranking learns
    P(item_a > item_b | context). This is more natural for implicit
    feedback where we only observe relative preferences.
    
    Final ranking is computed by aggregating pairwise comparisons
    (e.g., via Bradley-Terry model or simple win counting).
    
    Example::

        ranker = PairwiseRanker(
            compare_fn=my_pairwise_model.predict_proba
        )
        ranker.fit()
        ranked = ranker.rank(candidates, context)
    """
    
    def __init__(
        self,
        compare_fn: Optional[Callable[[Any, Any, Dict], float]] = None,
        aggregation: str = "wins",
        max_comparisons: Optional[int] = None,
        name: str = "pairwise",
    ):
        """
        Args:
            compare_fn: function(item_a, item_b, context) -> P(a > b)
            aggregation: how to aggregate comparisons:
                        'wins' - count wins
                        'bradley_terry' - fit BT model
            max_comparisons: limit comparisons for efficiency
            name: identifier for this ranker
        """
        super().__init__(name=name)
        
        self.compare_fn = compare_fn
        self.aggregation = aggregation
        self.max_comparisons = max_comparisons
    
    def fit(
        self,
        compare_fn: Optional[Callable] = None,
        **kwargs
    ) -> "PairwiseRanker":
        """
        Configure the ranker.
        
        Args:
            compare_fn: pairwise comparison function
        """
        if compare_fn is not None:
            self.compare_fn = compare_fn
        
        if self.compare_fn is None:
            raise ValueError("Must provide compare_fn")
        
        self._is_fitted = True
        return self
    
    def rank(
        self,
        candidates: Union[List[Candidate], RetrievalResult],
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RankingResult:
        """
        Rank candidates via pairwise comparisons.
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        candidates = self._candidates_to_list(candidates)
        context = context or {}
        n = len(candidates)
        
        if n == 0:
            return RankingResult(
                candidates=[],
                ranker_name=self.name,
                timing_ms=0,
            )
        
        # compute pairwise comparisons
        # win_matrix[i, j] = P(candidate_i > candidate_j)
        win_matrix = np.zeros((n, n))
        
        pairs = self._get_comparison_pairs(n)
        
        for i, j in pairs:
            prob_i_wins = self.compare_fn(
                candidates[i].item_id,
                candidates[j].item_id,
                context
            )
            win_matrix[i, j] = prob_i_wins
            win_matrix[j, i] = 1 - prob_i_wins
        
        # aggregate to scores
        if self.aggregation == "wins":
            scores = win_matrix.sum(axis=1)
        elif self.aggregation == "bradley_terry":
            scores = self._bradley_terry_scores(win_matrix)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # create ranked candidates
        ranked = []
        for idx in np.argsort(scores)[::-1]:
            ranked.append(RankedCandidate(
                item_id=candidates[idx].item_id,
                score=float(scores[idx]),
                retrieval_score=candidates[idx].score,
            ))
        
        # assign ranks
        for i, rc in enumerate(ranked):
            rc.rank = i + 1
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RankingResult(
            candidates=ranked,
            query_id=context.get('user_id'),
            ranker_name=self.name,
            timing_ms=elapsed,
        )
    
    def _get_comparison_pairs(self, n: int) -> List[Tuple[int, int]]:
        """Get pairs to compare, respecting max_comparisons limit."""
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        if self.max_comparisons is not None and len(all_pairs) > self.max_comparisons:
            # sample pairs
            indices = np.random.choice(
                len(all_pairs), 
                size=self.max_comparisons, 
                replace=False
            )
            return [all_pairs[i] for i in indices]
        
        return all_pairs
    
    def _bradley_terry_scores(
        self, 
        win_matrix: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Bradley-Terry model scores from win matrix.
        
        Iterative algorithm to find item strengths that best
        explain the observed win probabilities.
        """
        n = win_matrix.shape[0]
        scores = np.ones(n)  # initial strengths
        
        for _ in range(max_iter):
            old_scores = scores.copy()
            
            for i in range(n):
                # update score for item i
                wins = win_matrix[i].sum()
                denom = 0
                for j in range(n):
                    if i != j:
                        denom += 1 / (scores[i] + scores[j])
                
                if denom > 0:
                    scores[i] = wins / denom
            
            # normalize
            scores = scores / scores.sum() * n
            
            # check convergence
            if np.abs(scores - old_scores).max() < tol:
                break
        
        return scores
