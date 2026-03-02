"""
Traditional ML Algorithms
=========================

Classical machine learning algorithms for content-based recommendations.

This module provides:
- TF-IDF based recommendations
- SVM (Support Vector Machines)
- LightGBM (Gradient Boosting)
- Decision Trees
- Logistic Regression
- Vowpal Wabbit

Usage:
------
    from corerec.engines.content import traditional
    
    # Traditional ML models
    model = traditional.TFIDFRecommender()
    model = traditional.SVMRecommender()
    model = traditional.LightGBMRecommender()
    model = traditional.DecisionTreeRecommender()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Export Traditional ML Algorithms
# ============================================================================

try:
    from .tfidf import TFIDFTraditional
except ImportError:
    TFIDFTraditional = None

try:
    from .svm import SVMRecommender
except ImportError:
    SVMRecommender = None

try:
    from .lightgbm import LightGBMRecommender
except ImportError:
    LightGBMRecommender = None

try:
    from .decision_tree import DecisionTreeRecommender
except ImportError:
    DecisionTreeRecommender = None

try:
    from .LR import LogisticRegressionRecommender
except ImportError:
    LogisticRegressionRecommender = None

try:
    from .vw import VowpalWabbitRecommender
except ImportError:
    VowpalWabbitRecommender = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    "TFIDFTraditional",
    "SVMRecommender",
    "LightGBMRecommender",
    "DecisionTreeRecommender",
    "LogisticRegressionRecommender",
    "VowpalWabbitRecommender",
]
