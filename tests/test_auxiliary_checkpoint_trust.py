import pytest
import torch

from corerec.api.exceptions import SaveLoadError
from corerec.core.base_model import BaseModel
from corerec.trainer.trainer import Trainer
from corerec.hybrid.retrieval_then_rerank import RetrievalThenRerank
from tests.test_persistence_security import Payload


class TinyModel(BaseModel):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.linear = torch.nn.Linear(config['width'], 1)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize('name', ['model', 'trainer', 'hybrid', 'transformer'])
def test_auxiliary_loaders_reject_untrusted_pickle(name, tmp_path):
    if name == 'transformer':
        pytest.importorskip('transformers')
        pytest.importorskip('torchvision')
        from corerec.experimental.towers.transformer_tower import TransformerTower
        load = TransformerTower.load
    elif name == 'trainer':
        trainer = Trainer.__new__(Trainer)
        trainer.device = 'cpu'
        load = trainer.load_checkpoint
    else:
        load = BaseModel.load if name == 'model' else RetrievalThenRerank.load
    marker = tmp_path / 'executed'
    path = tmp_path / 'untrusted.pt'
    torch.save(Payload(marker), path)
    with pytest.raises(SaveLoadError, match='allow_pickle=True'):
        load(path)
    assert not marker.exists()


def test_trusted_base_model_checkpoint_restores_weights(tmp_path):
    model = TinyModel('tiny', {'width': 2})
    path = model.save(str(tmp_path))
    with pytest.warns(UserWarning, match='arbitrary code'):
        loaded = TinyModel.load(path, device=torch.device('cpu'), allow_pickle=True)
    torch.testing.assert_close(loaded(torch.ones(3, 2)), model(torch.ones(3, 2)))


def test_trusted_trainer_checkpoint_restores_training_state(tmp_path):
    model = TinyModel('tiny', {'width': 2})
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    trainer = Trainer(model, optimizer, device=torch.device('cpu'),
                      checkpoint_dir=str(tmp_path / 'checkpoints'), log_dir=str(tmp_path / 'logs'))
    optimizer.zero_grad()
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    trainer.history['train_loss'] = [.2]
    trainer.save_checkpoint(1)
    path = next((tmp_path / 'checkpoints').glob('*.pt'))
    expected = model(torch.ones(2, 2)).detach().clone()
    with torch.no_grad():
        model.linear.weight.zero_()
    trainer.history = {}
    with pytest.warns(UserWarning, match='arbitrary code'):
        trainer.load_checkpoint(str(path), allow_pickle=True)
    torch.testing.assert_close(model(torch.ones(2, 2)), expected)
    assert trainer.history['train_loss'] == [.2]
    assert optimizer.state
