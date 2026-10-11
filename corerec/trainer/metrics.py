"""
Metrics module for CoreRec framework.

This module provides implementations of common metrics for recommendation systems.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable


def precision_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Compute precision@k for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Precision@k score
    """
    # Get top-k predicted items
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # Compute precision for each row
    precision = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        precision[i] = torch.sum(labels[i, top_k_indices[i]]) / k

    # Return mean precision
    return torch.mean(precision)


def recall_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Compute recall@k for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Recall@k score
    """
    # Get top-k predicted items
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # Compute recall for each row
    recall = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        if torch.sum(labels[i]) > 0:
            recall[i] = torch.sum(labels[i, top_k_indices[i]]) / torch.sum(labels[i])

    # Return mean recall
    return torch.mean(recall)


def mean_reciprocal_rank(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute Mean Reciprocal Rank (MRR) for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]

    Returns:
        torch.Tensor: MRR score
    """
    # Get rank indices
    _, indices = torch.sort(predictions, descending=True, dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # MRR is 1 / rank of the *first* relevant item; this averaged over all of
    # them, disagreeing with corerec.evaluation's mrr_at_k
    mrr = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        hits = torch.nonzero(labels[i, indices[i]] > 0, as_tuple=True)[0]
        if hits.numel() > 0:
            mrr[i] = 1.0 / (hits[0].float() + 1)

    # Return mean MRR
    return torch.mean(mrr)


def ndcg_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Compute Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: NDCG@k score
    """
    # Get top-k predicted items
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # Compute DCG and IDCG for each row
    ndcg = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        # Get relevance scores for top-k items
        relevance = labels[i, top_k_indices[i]]

        # Compute DCG
        discounts = torch.log2(
            torch.arange(2, len(relevance) + 2, dtype=torch.float, device=labels.device)
        )
        dcg = torch.sum(relevance / discounts)

        # Compute IDCG (ideal DCG)
        ideal_relevance, _ = torch.sort(labels[i], descending=True)
        ideal_relevance = ideal_relevance[:k]
        idcg = torch.sum(ideal_relevance / discounts[: len(ideal_relevance)])

        # Compute NDCG
        if idcg > 0:
            ndcg[i] = dcg / idcg

    # Return mean NDCG
    return torch.mean(ndcg)


def hit_rate_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Compute Hit Rate at k for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: Hit Rate@k score
    """
    # Get top-k predicted items
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # Compute hit rate for each row
    hit_rate = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        # Check if any of the top-k items are relevant
        if torch.sum(labels[i, top_k_indices[i]]) > 0:
            hit_rate[i] = 1.0

    # Return mean hit rate
    return torch.mean(hit_rate)


def average_precision_at_k(
    predictions: torch.Tensor, labels: torch.Tensor, k: int = 10
) -> torch.Tensor:
    """Compute Average Precision at k for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores or indices of shape [batch_size, n_items]
        labels (torch.Tensor): Ground truth labels of shape [batch_size, n_items] or [batch_size]
        k (int): Number of top items to consider

    Returns:
        torch.Tensor: AP@k score
    """
    # Get top-k predicted items
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.shape[1]), dim=1)

    # Prepare labels for comparison
    if labels.dim() == 1:
        # Convert to one-hot encoding
        device = labels.device
        batch_size = labels.shape[0]
        n_items = predictions.shape[1]
        label_matrix = torch.zeros(batch_size, n_items, device=device)
        label_matrix[torch.arange(batch_size, device=device), labels] = 1
        labels = label_matrix

    # Compute AP for each row
    ap = torch.zeros(labels.shape[0], device=labels.device)
    for i in range(labels.shape[0]):
        # Get relevance scores for top-k items
        relevance = labels[i, top_k_indices[i]]

        if torch.sum(relevance) > 0:
            # precision at each of the top positions (len(relevance) = min(k, n_items);
            # a fixed length k crashed on catalogues smaller than k)
            ranks = torch.arange(1, len(relevance) + 1, device=labels.device, dtype=torch.float)
            precision_at_j = torch.cumsum(relevance, 0) / ranks
            # normalise by what could have been found, as corerec.evaluation does;
            # dividing by the hits made one lucky hit out of five relevant score 1.0
            n_relevant = int((labels[i] > 0).sum())
            ap[i] = torch.sum(precision_at_j * relevance) / min(k, n_relevant)

    # Return mean AP
    return torch.mean(ap)


def binary_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute binary accuracy for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores of shape [batch_size]
        labels (torch.Tensor): Ground truth labels of shape [batch_size]

    Returns:
        torch.Tensor: Binary accuracy score
    """
    # Convert predictions to binary
    binary_preds = (torch.sigmoid(predictions) > 0.5).float()

    # Compute accuracy
    correct = (binary_preds == labels).float()
    accuracy = torch.mean(correct)

    return accuracy


def auc_roc(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute AUC-ROC for recommendations.

    This implementation uses a simple approximation of AUC-ROC.

    Args:
        predictions (torch.Tensor): Predicted scores of shape [batch_size]
        labels (torch.Tensor): Ground truth labels of shape [batch_size]

    Returns:
        torch.Tensor: AUC-ROC score
    """
    from scipy.stats import rankdata

    # Mann-Whitney U with average ranks, so tied scores count half. The old
    # sort-then-trapezoid result depended on how ties happened to be ordered
    # (a constant predictor could score anywhere from 0 to 1, not 0.5).
    y = labels.detach().cpu().numpy() > 0
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5, device=labels.device)
    ranks = rankdata(predictions.detach().cpu().numpy())
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return torch.tensor(float(auc), device=labels.device)


def f1_score(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute F1 score for recommendations.

    Args:
        predictions (torch.Tensor): Predicted scores of shape [batch_size]
        labels (torch.Tensor): Ground truth labels of shape [batch_size]

    Returns:
        torch.Tensor: F1 score
    """
    # Convert predictions to binary
    binary_preds = (torch.sigmoid(predictions) > 0.5).float()

    # Compute true positives, false positives, and false negatives
    tp = torch.sum((binary_preds == 1) & (labels == 1))
    fp = torch.sum((binary_preds == 1) & (labels == 0))
    fn = torch.sum((binary_preds == 0) & (labels == 1))

    # Compute precision and recall
    precision = tp / (tp + fp) if tp + fp > 0 else torch.tensor(0.0, device=labels.device)
    recall = tp / (tp + fn) if tp + fn > 0 else torch.tensor(0.0, device=labels.device)

    # Compute F1 score
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else torch.tensor(0.0, device=labels.device)
    )

    return f1


def get_metrics_dict(k: int = 10) -> Dict[str, Callable]:
    """Get a dictionary of common metrics for recommendation systems.

    Args:
        k (int): Top-k for metrics that require it

    Returns:
        Dict[str, Callable]: Dictionary of metrics
    """
    return {
        "precision": lambda p, l: precision_at_k(p, l, k),
        "recall": lambda p, l: recall_at_k(p, l, k),
        "mrr": mean_reciprocal_rank,
        "ndcg": lambda p, l: ndcg_at_k(p, l, k),
        "hit_rate": lambda p, l: hit_rate_at_k(p, l, k),
        "map": lambda p, l: average_precision_at_k(p, l, k),
        "accuracy": binary_accuracy,
        "auc": auc_roc,
        "f1": f1_score,
    }
