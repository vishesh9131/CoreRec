"""
Sandbox - Experimental & Development Methods
=============================================

This module contains 80+ recommendation methods under active development.
Methods here are functional but not yet production-ready.

Organization:
-------------
- collaborative/  - 45+ collaborative filtering methods
- content_based/  - 35+ content-based methods

Why Sandbox?
------------
Keeping experimental methods separate ensures:
1. Main engine stays lean and tested
2. Clear distinction between stable vs experimental
3. Freedom to iterate without breaking production code
4. Gradual graduation of methods to main engine

Usage:
------
    from corerec.sandbox.collaborative import DeepFM, DCN
    from corerec.sandbox.content_based import CNN, Transformer

Status Levels:
--------------
- Alpha: Early development, API may change
- Beta: Feature complete, needs testing
- Stable: Ready for graduation to main engine

Check each method's docstring for status.

Contributing:
-------------
When a sandbox method is production-ready:
1. Add comprehensive tests
2. Document edge cases
3. Optimize performance
4. Create migration guide
5. Graduate to main engine

Author: Vishesh Yadav
"""

from . import collaborative
from . import content_based

__all__ = ["collaborative", "content_based"]


def list_all_methods():
    """List all sandbox methods."""
    return {
        "collaborative": [
            "Matrix Factorization: SVD, ALS, NMF, PMF, BPR",
            "Neural: DeepFM, DCN, AutoInt, PNN, xDeepFM, DLRM",
            "Sequential: GRU, LSTM, Caser, SASRec, NextItNet, TiSAS",
            "Attention: DIEN, DIN, BST, DMR",
            "Graph: DeepWalk, GNN variants",
            "Variational: VAE, BiVAE, CVAE",
            "Bayesian: Bayesian MF, MCMC",
            "Multi-Task: MMOE, PLE, ESMM",
            "Others: RLRMC, SLI, SUM, RBM, GeoMLC",
        ],
        "content_based": [
            "Traditional ML: SVM, LightGBM, Decision Trees, VW",
            "Neural: CNN, RNN, Transformer, VAE, Autoencoder",
            "Graph: GNN, Semantic Models",
            "Hybrid: Ensemble, Attention",
            "Context: User profiling, Item profiling",
            "Fairness: Fair ranking, Explainable AI",
            "Learning: Transfer, Meta, Few-shot, Zero-shot",
            "Multi-Modal: Fusion strategies",
            "Others: Doc2Vec, TDM, MIND, AITM, LSTUR, NAML",
        ],
    }


def get_graduation_queue():
    """List methods close to graduation."""
    return [
        "DeepFM - Nearly ready, needs final testing",
        "DCN - Performance optimizations in progress",
        "SASRec - Already good, considering graduation",
        "CNN - Needs multi-modal integration",
        "Transformer - Waiting for BERT4Rec to stabilize",
    ]

