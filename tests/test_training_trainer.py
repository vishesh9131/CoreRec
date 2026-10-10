"""corerec.training: the generic Trainer and its callbacks (#79: was ~20% covered)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from corerec.training import EarlyStopping, LearningRateScheduler, ModelCheckpoint, Trainer


def _data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    return DataLoader(TensorDataset(x, y), batch_size=16)


def _trainer(callbacks=(), lr=0.05):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return Trainer(model, opt, nn.MSELoss(), callbacks=list(callbacks), device="cpu"), model


def test_training_lowers_the_loss():
    trainer, model = _trainer()
    dl = _data()
    before = trainer._validate_epoch(dl)
    trainer.train(dl, val_loader=dl, epochs=20)
    assert trainer._validate_epoch(dl) < before / 10


def test_batches_must_be_input_target_pairs():
    trainer, _ = _trainer()
    bad = DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=4)  # 1-tuples
    with pytest.raises(ValueError, match="Batch format not recognized"):
        trainer.train(bad, epochs=1)


def test_early_stopping_ends_training_when_the_metric_stalls():
    stop = EarlyStopping(patience=2, monitor="val_loss", min_delta=1e9)  # nothing counts
    trainer, _ = _trainer([stop])
    dl = _data()
    trainer.train(dl, val_loader=dl, epochs=50)
    assert stop.stop_training and stop.stopped_epoch == 2  # 3 epochs, not 50


def test_model_checkpoint_writes_a_loadable_state_dict(tmp_path):
    """It printed 'Saved model to ...' but never wrote anything."""
    path = tmp_path / "ckpt" / "best.pt"
    trainer, model = _trainer([ModelCheckpoint(str(path), monitor="val_loss")])
    dl = _data()
    trainer.train(dl, val_loader=dl, epochs=3)
    assert path.exists()
    restored = nn.Linear(4, 1)
    restored.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(5, 4)
    assert torch.allclose(restored(x), model(x))  # loss falls each epoch, so last = best


def test_model_checkpoint_needs_a_model_outside_trainer(tmp_path):
    cb = ModelCheckpoint(str(tmp_path / "m.pt"), save_best_only=False)
    with pytest.raises(RuntimeError, match="no model"):
        cb.on_epoch_end(0, {})


def test_learning_rate_schedule_is_applied_each_epoch():
    trainer, _ = _trainer([LearningRateScheduler(lambda epoch: 0.1 / (epoch + 1), verbose=False)])
    trainer.train(_data(), epochs=3)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 / 3)
