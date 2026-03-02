"""
Explanation Module - Recommendation Explanations

Generate human-readable explanations for why items were recommended.
Essential for user trust and debugging.

Usage:
    from corerec.explanation import FeatureExplainer, GenerativeExplainer
    
    explainer = FeatureExplainer()
    explanation = explainer.explain(item, user_context)
"""

from .base import BaseExplainer, Explanation
from .feature_based import FeatureExplainer
from .generative import GenerativeExplainer

__all__ = [
    "BaseExplainer",
    "Explanation",
    "FeatureExplainer",
    "GenerativeExplainer",
]
