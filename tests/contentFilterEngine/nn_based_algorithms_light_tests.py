import unittest
import torch
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

NN_BASE = Path(__file__).resolve().parents[2] / "corerec" / "engines" / "contentFilterEngine" / "nn_based_algorithms"


def load(fname: str):
    spec = spec_from_file_location(fname, str(NN_BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestNNBasedAlgorithmsLight(unittest.TestCase):
    def test_rnn_forward(self):
        rnn_mod = load("rnn.py")
        model = rnn_mod.RNNModel(input_dim=6, embed_dim=8, hidden_dim=10, num_layers=1, bidirectional=False, num_classes=3)
        x = torch.randn(4, 5, 6)
        y = model(x)
        self.assertEqual(y.shape, (4, 3))

    def test_transformer_forward(self):
        tr_mod = load("transformer.py")
        model = tr_mod.TransformerModel(input_dim=6, embed_dim=8, num_heads=2, hidden_dim=16, num_layers=1, num_classes=3)
        x = torch.randn(4, 5, 6)
        y = model(x)
        self.assertEqual(y.shape, (4, 3))

    def test_cnn_forward(self):
        cnn_mod = load("cnn.py")
        model = cnn_mod.CNN(input_dim=6, num_classes=2, emb_dim=8, kernel_sizes=[3, 3, 3], num_filters=4)
        x = torch.randn(4, 6)
        y = model(x)
        self.assertEqual(y.shape, (4, 2))

    def test_autoencoder_forward(self):
        ae_mod = load("autoencoder.py")
        model = ae_mod.Autoencoder(input_dim=6, hidden_dim=8, latent_dim=4)
        x = torch.randn(4, 6)
        y = model(x)
        self.assertEqual(y.shape, (4, 6))


if __name__ == "__main__":
    unittest.main(verbosity=2) 