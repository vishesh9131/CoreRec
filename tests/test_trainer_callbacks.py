"""corerec.trainer callbacks driven by the real Trainer (#79)."""

import torch
import torch.nn as nn
import pytest

from corerec.core.base_model import BaseModel
from corerec.trainer import EarlyStopping, ModelCheckpoint, Trainer


class Reg(BaseModel):
    def __init__(self):
        super().__init__("reg", {})
        self.lin = nn.Linear(3, 1)

    def forward(self, x):
        return self.lin(x)

    def train_step(self, batch, optimizer):
        loss = nn.functional.mse_loss(self(batch["x"]), batch["y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validate_step(self, batch):
        with torch.no_grad():
            return {"val_loss": nn.functional.mse_loss(self(batch["x"]), batch["y"]).item()}


def _batches(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(8, 3, generator=g)
        out.append({"x": x, "y": x @ torch.tensor([[1.0], [2.0], [-1.0]])})
    return out


def _trainer(tmp_path, callbacks):
    torch.manual_seed(0)
    model = Reg()
    return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.05), device=torch.device("cpu"),
                   callbacks=callbacks, checkpoint_dir=str(tmp_path / "ckpt"),
                   log_dir=str(tmp_path / "logs")), model


def test_checkpoint_with_validation_writes_weights_to_the_formatted_path(tmp_path):
    """filepath.format(**train, **val) raised: both dicts carry 'steps'."""
    cb = ModelCheckpoint(str(tmp_path / "w_{epoch}_{val_loss:.3f}.pt"), save_weights_only=True)
    trainer, model = _trainer(tmp_path, [cb])
    data = _batches()
    trainer.train(data, val_loader=data, epochs=2)
    files = sorted(p.name for p in tmp_path.glob("w_*.pt"))
    assert [f.split("_")[1] for f in files] == ["1", "2"]
    state = torch.load(tmp_path / files[-1], weights_only=True)
    assert torch.equal(state["lin.weight"], model.lin.weight)


def test_full_checkpoint_goes_to_filepath_not_checkpoint_dir(tmp_path):
    target = tmp_path / "best.pt"
    trainer, _ = _trainer(tmp_path, [ModelCheckpoint(str(target), save_best_only=True)])
    data = _batches()
    trainer.train(data, val_loader=data, epochs=2, save_freq=100)  # trainer's own saves off
    assert target.exists()
    assert not list((tmp_path / "ckpt").glob("model_epoch_*"))  # the callback wrote here before
    assert torch.load(target, weights_only=False)["epoch"] == 2  # loss falls, so last = best


def test_save_every_epoch_without_validation(tmp_path):
    """With save_best_only=False it needs no metric; it used to save nothing without val."""
    trainer, _ = _trainer(tmp_path, [ModelCheckpoint(str(tmp_path / "e{epoch}.pt"),
                                                     save_weights_only=True)])
    trainer.train(_batches(), epochs=3)
    assert sorted(p.name for p in tmp_path.glob("e*.pt")) == ["e1.pt", "e2.pt", "e3.pt"]


def test_best_value_resets_between_runs(tmp_path):
    cb = ModelCheckpoint(str(tmp_path / "b.pt"), save_best_only=True, save_weights_only=True)
    trainer, _ = _trainer(tmp_path, [cb])
    data = _batches()
    trainer.train(data, val_loader=data, epochs=1)
    cb.best_value = -1.0  # as if a previous run had an unbeatable best
    (tmp_path / "b.pt").unlink()
    trainer.train(data, val_loader=data, epochs=1)
    assert (tmp_path / "b.pt").exists()


def test_early_stopping_stops_on_a_stalled_metric(tmp_path):
    stop = EarlyStopping(patience=2, min_delta=1e9)  # no epoch counts as better
    trainer, _ = _trainer(tmp_path, [stop])
    data = _batches()
    trainer.train(data, val_loader=data, epochs=30)
    assert stop.stopped_epoch == 2
