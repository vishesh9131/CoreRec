import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_basic_trainer_imports_without_tracking_packages(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mlflow', None)
    monkeypatch.setitem(sys.modules, 'wandb', None)
    callbacks = importlib.import_module('corerec.trainer.callbacks')
    package = importlib.reload(importlib.import_module('corerec.trainer'))
    assert package.Trainer is not None
    assert package.Callback is callbacks.Callback
    assert package.EarlyStopping is callbacks.EarlyStopping
    assert callbacks.EarlyStopping() is not None
    with pytest.raises(ImportError, match='mlflow'):
        callbacks.MLflowLogger()
    with pytest.raises(ImportError, match='wandb'):
        callbacks.WandbLogger()


@pytest.mark.parametrize('name,module', [('MLflowLogger', 'mlflow'), ('WandbLogger', 'wandb')])
def test_tracking_callbacks_still_call_selected_backend(name, module, monkeypatch):
    backend = Mock()
    monkeypatch.setitem(sys.modules, module, backend)
    callbacks = importlib.import_module('corerec.trainer.callbacks')
    callback = getattr(callbacks, name)()
    trainer = SimpleNamespace(model=torch.nn.Linear(2, 1))
    callback.on_train_begin(trainer)
    callback.on_epoch_end(trainer, 1, {'loss': .2}, {'loss': .3})
    callback.on_train_end(trainer)
    if module == 'mlflow':
        backend.start_run.assert_called_once()
        backend.log_metrics.assert_called_once()
        backend.end_run.assert_called_once()
    else:
        backend.init.assert_called_once()
        backend.log.assert_called_once()
        backend.finish.assert_called_once()
