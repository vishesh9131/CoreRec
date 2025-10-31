"""
Graph-Based Algorithms
======================

Graph neural networks and graph-based methods for content recommendations.

This module provides:
- GNN (Graph Neural Networks)
- Semantic Models
- Graph Filtering

Usage:
------
    from corerec.engines.content import graph
    
    # Graph-based models
    model = graph.GNNContentFilter()
    model = graph.GraphFiltering()
    model = graph.SemanticModels()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Export Graph-Based Algorithms
# ============================================================================

try:
    from .gnn import GNNContentFilter
except ImportError:
    GNNContentFilter = None

try:
    from .graph_filtering import GraphFiltering
except ImportError:
    GraphFiltering = None

try:
    from .semantic_models import SemanticModels
except ImportError:
    SemanticModels = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    'GNNContentFilter',
    'GraphFiltering',
    'SemanticModels',
]
