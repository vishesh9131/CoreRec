"""
CoreRec Evaluation Framework

Comprehensive metrics and evaluation tools for recommendation systems.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from corerec.evaluation.metrics import RankingMetrics, ClassificationMetrics, DiversityMetrics
from corerec.evaluation.evaluator import Evaluator, CrossValidator

__all__ = [
    "RankingMetrics",
    "ClassificationMetrics",
    "DiversityMetrics",
    "Evaluator",
    "CrossValidator",
]

