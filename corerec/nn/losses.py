"""
Ranking losses over a [batch, 1 + K] score matrix.

Column 0 is the positive item, columns 1..K are sampled negatives -- the shape
corerec.nn.Recommender hands to the loss. Any callable with this signature can
be passed as ``Recommender(loss=...)``.
"""

import torch
import torch.nn.functional as F


def bpr_loss(scores: torch.Tensor) -> torch.Tensor:
    """Bayesian personalised ranking: the positive should outscore each negative."""
    return -F.logsigmoid(scores[:, :1] - scores[:, 1:]).mean()


def bce_loss(scores: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy: positive -> 1, negatives -> 0."""
    pos = F.binary_cross_entropy_with_logits(scores[:, 0], torch.ones_like(scores[:, 0]))
    neg = F.binary_cross_entropy_with_logits(scores[:, 1:], torch.zeros_like(scores[:, 1:]))
    return pos + neg


def sampled_softmax_loss(scores: torch.Tensor) -> torch.Tensor:
    """Softmax cross-entropy of the positive against the sampled negatives."""
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)


LOSSES = {"bpr": bpr_loss, "bce": bce_loss, "sampled_softmax": sampled_softmax_loss}

__all__ = ["bpr_loss", "bce_loss", "sampled_softmax_loss", "LOSSES"]
