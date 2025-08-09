import unittest
import sys
from types import ModuleType
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import torch

BASE = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / "learning_paradigms"


def ensure_module(name: str):
    m = ModuleType(name)
    sys.modules[name] = m
    return m


def load(filename: str):
    spec = spec_from_file_location(filename, str(BASE / filename))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestLearningParadigms(unittest.TestCase):
    def setUp(self):
        # stub corerec.train and corerec.predict
        ensure_module("corerec")
        tr = ensure_module("corerec.train")
        pr = ensure_module("corerec.predict")
        self.train_called = {}
        def train_model(model, loader, criterion, optimizer, epochs):
            self.train_called.update({"model": model, "epochs": epochs})
        def predict(model, graph, node_index, top_k, threshold):
            return list(range(min(top_k, 3)))
        tr.train_model = train_model
        pr.predict = predict

    def test_zero_shot_predict(self):
        mod = load("zero_shot.py")
        model = torch.nn.Linear(4, 2)
        zs = mod.ZeroShotLearner(model)
        out = zs.predict(graph=None, node_index=0, top_k=5, threshold=0.5)
        self.assertEqual(out, [0, 1, 2])

    def test_transfer_learning_train_and_predict(self):
        mod = load("transfer_learning.py")
        model = torch.nn.Linear(4, 2)
        tl = mod.TransferLearningLearner(model, data_loader=None, criterion=None, optimizer=None, num_epochs=2)
        tl.train()
        self.assertEqual(self.train_called.get("epochs"), 2)
        out = tl.predict(graph=None, node_index=1, top_k=2, threshold=0.1)
        self.assertEqual(out, [0, 1])

    def test_meta_learning_train_and_predict(self):
        mod = load("meta_learning.py")
        model = torch.nn.Linear(4, 2)
        ml = mod.MetaLearner(model, data_loader=None, criterion=None, optimizer=None, num_epochs=3)
        ml.train()
        self.assertEqual(self.train_called.get("epochs"), 3)
        out = ml.predict(graph=None, node_index=2, top_k=1, threshold=0.9)
        self.assertEqual(out, [0])


if __name__ == "__main__":
    unittest.main(verbosity=2) 