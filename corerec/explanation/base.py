"""
Base classes for explanation generation.

Explanations help users understand why items were recommended,
increasing trust and engagement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Explanation:
    """
    An explanation for a recommendation.
    
    Attributes:
        item_id: the item being explained
        text: human-readable explanation
        explanation_type: category of explanation
        confidence: how confident we are in this explanation
        supporting_items: items that support this explanation
        features: key features that drove the recommendation
    """
    item_id: Any
    text: str
    explanation_type: str = "generic"
    confidence: float = 1.0
    supporting_items: List[Any] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Explanation({self.explanation_type}: {self.text[:50]}...)"


class BaseExplainer(ABC):
    """
    Abstract base class for explanation generators.
    
    Explainers take recommendation context and generate
    human-readable explanations for why items were recommended.
    
    Common explanation types:
    - "collaborative": "Users like you also bought..."
    - "content": "Because you liked similar items..."
    - "popularity": "Trending in your area..."
    - "personalized": "Based on your preferences..."
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def explain(
        self,
        item_id: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> Explanation:
        """
        Generate an explanation for a recommended item.
        
        Args:
            item_id: the item to explain
            context: user context, recommendation context, etc.
            **kwargs: explainer-specific parameters
        
        Returns:
            Explanation object
        """
        pass
    
    def explain_batch(
        self,
        item_ids: List[Any],
        context: Dict[str, Any],
        **kwargs
    ) -> List[Explanation]:
        """
        Generate explanations for multiple items.
        
        Default implementation loops; subclasses can optimize.
        """
        return [self.explain(item_id, context, **kwargs) for item_id in item_ids]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
