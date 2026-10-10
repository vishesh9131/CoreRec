"""The training loop DCN and DeepFM share (#77).

Both score (user, item[, side features]) rows with BCE for implicit
feedback or MSE for ratings, and each carried its own copy of this loop.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def train_pointwise(model: nn.Module, X, y, *, task: str, epochs: int, batch_size: int,
                    learning_rate: float, device, verbose: bool = False, name: str = "model") -> None:
    """Train ``model`` on rows ``X`` [N, fields] (int) against labels ``y`` [N].

    A fresh shuffle each epoch on ``device``; batches of one are skipped
    (BatchNorm needs two). Warns if the trained scores collapse to a constant.
    """
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.long).to(device)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32)).to(device)
    n = X_t.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss() if task != "rating" else nn.MSELoss()

    model.train()
    n_batches = (n + batch_size - 1) // batch_size
    for epoch in range(epochs):
        total = 0.0
        perm = torch.randperm(n, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if idx.numel() < 2:
                continue
            loss = criterion(model(X_t[idx]), y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if verbose:
            logger.info("%s epoch %d/%d, loss %.4f", name, epoch + 1, epochs, total / n_batches)

    # a healthy model scores items differently; near-constant output means the
    # labels don't match the task and every ranking from it is meaningless
    model.eval()
    with torch.no_grad():
        score_std = float(model(X_t[: min(2048, n)]).std().item())
    if score_std < 1e-4:
        logger.warning(
            "%s output collapsed (score std=%.2e): predictions are nearly constant, so "
            "rankings will be meaningless. Check that labels match the task ('implicit' "
            "expects relevance, 'rating' expects scores).", name, score_std)
