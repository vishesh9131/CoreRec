"""
Build new recommender models in plain PyTorch, with CoreRec's parts.

    import torch.nn as nn
    from corerec.nn import Recommender, SASRecBlock, causal_mask, bpr_loss

Write an nn.Module with forward(query, items) -> scores, wrap it in
Recommender, and it trains, recommends, saves, serves (ModelServer), evaluates
(Evaluator) and exports (corerec.export.to_onnx) like any built-in model.
See corerec.nn.recommender for the contract and corerec.nn.models for templates.
"""

from corerec.nn.layers import (CrossLayer, FMInteraction, HSTUBlock, MLP, SASRecBlock,
                               causal_mask)
from corerec.nn.losses import bce_loss, bpr_loss, sampled_softmax_loss
from corerec.nn.models import HSTUTransformer, MatrixFactorization, SequentialTransformer
from corerec.nn.recommender import Recommender

__all__ = [
    "Recommender",
    "MatrixFactorization", "SequentialTransformer", "HSTUTransformer",
    "HSTUBlock", "SASRecBlock", "CrossLayer", "FMInteraction", "MLP", "causal_mask",
    "bpr_loss", "bce_loss", "sampled_softmax_loss",
]
