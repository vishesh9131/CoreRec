"""corerec.trainer.parameter_sync, in a one-process gloo group (#79)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from corerec.trainer.parameter_sync import GradientAccumulator, ParameterSync


@pytest.fixture
def group():
    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_sparse_embeddings_are_found_and_synced(group, monkeypatch):
    """getattr(param, 'sparse') was always False, so nothing was ever synced."""
    model = nn.Sequential(nn.Embedding(10, 4, sparse=True), nn.Linear(4, 1),
                          nn.Embedding(5, 4))  # dense table: not synced
    sync = ParameterSync(model, rank=0, world_size=1, sync_interval=2)
    assert sync.sparse_param_names == ["0.weight"]

    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda t, op=None: calls.append(t.shape))
    sync.sync()
    assert calls == []                 # step 1: not yet
    sync.sync()
    assert calls == [(10, 4)]          # step 2: synced


def test_gradient_accumulation_matches_one_big_batch():
    torch.manual_seed(0)
    x, y = torch.randn(8, 3), torch.randn(8, 1)
    big, acc = nn.Linear(3, 1), nn.Linear(3, 1)
    acc.load_state_dict(big.state_dict())

    nn.functional.mse_loss(big(x), y).backward()

    ga = GradientAccumulator(acc, accumulation_steps=2)
    for half in (slice(0, 4), slice(4, 8)):
        nn.functional.mse_loss(acc(x[half]), y[half]).backward()
        ga.accumulate_gradients()
    ga.apply_accumulated_gradients()
    for p, q in zip(big.parameters(), acc.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
    assert ga.current_step == 0
