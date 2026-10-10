"""corerec.trainer.OnlineTrainer: background updates from queued data (#79)."""

import logging
import time

import pandas as pd
import torch
import torch.nn as nn

from corerec.core.base_model import BaseModel
from corerec.trainer import OnlineTrainer


class Bias(BaseModel):
    def __init__(self):
        super().__init__("bias", {})
        self.b = nn.Parameter(torch.zeros(1))

    def train_step(self, batch, optimizer):
        loss = ((self.b - batch["rating"]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}


def _trainer(tmp_path, **config):
    model = Bias()
    cfg = {"min_batch_size": 1, "max_wait_time": 0.05, "checkpoint_interval": 2, **config}
    return OnlineTrainer(model, torch.optim.SGD(model.parameters(), lr=0.5), torch.device("cpu"),
                         cfg, checkpoint_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs")), model


def test_queued_data_trains_the_model_in_the_background(tmp_path):
    trainer, model = _trainer(tmp_path)
    trainer.start_training()
    for _ in range(4):
        trainer.add_training_data_from_dataframe(
            pd.DataFrame({"user_id": [1, 2], "item_id": [3, 4], "rating": [2.0, 2.0]}))
        time.sleep(0.2)
    deadline = time.time() + 5
    while trainer.get_metrics()["updates"] < 2 and time.time() < deadline:
        time.sleep(0.05)
    trainer.stop_training_thread()

    m = trainer.get_metrics()
    assert m["train_steps"] >= 2 and m["queue_size"] == 0
    assert abs(model.b.item() - 2.0) < 1.0           # moved toward the ratings
    assert list((tmp_path / "ckpt").glob("online_model_*.pt"))  # checkpoint_interval=2


def test_trainers_sharing_a_log_dir_do_not_stack_handlers(tmp_path):
    """Each instance added a FileHandler to the shared logger: N copies of every line."""
    logger = logging.getLogger("OnlineTrainer")
    before = len(logger.handlers)
    _trainer(tmp_path)
    _trainer(tmp_path)
    assert len(logger.handlers) == before + 1
    for h in logger.handlers[before:]:
        logger.removeHandler(h)
        h.close()
