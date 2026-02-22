import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

BASE = (
    Path(__file__).resolve().parents[2]
    / "corerec"
    / "engines"
    / "content_based"
    / "multi_modal_cross_domain_methods"
)


def load(fname: str):
    spec = spec_from_file_location(fname, str(BASE / fname))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class DummyVectorizer:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, X):
        arr = np.ones((len(X), self.out_dim), dtype=float)

        class Wrapper:
            def __init__(self, a):
                self._a = a

            def toarray(self):
                return self._a

        return Wrapper(arr)


class DummyBinarizer:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, X):
        return np.ones((len(X), self.out_dim), dtype=float)


class TestMultiModalCrossDomain(unittest.TestCase):
    def test_multi_modal_forward(self):
        mm = load("multi_modal.py").MULTI_MODAL(
            text_model=DummyVectorizer(4),
            genre_model=DummyBinarizer(3),
        )
        feats = mm.forward(["a", "b"], [["g"], ["g"]])
        self.assertEqual(feats.shape, (2, 7))

    def test_cross_domain_transfer_eval(self):
        md = load("cross_domain.py")

        class Src(nn.Module):
            def forward(self, x):
                return x

        class Tgt(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.lin = nn.Linear(in_dim, 1)

            def forward(self, x):
                return self.lin(x).squeeze(-1)

        in_dim = 3
        data = [(torch.randn(5, in_dim), torch.randn(5))]
        tgt = Tgt(in_dim)
        cd = md.CROSS_DOMAIN(Src(), tgt)
        optimizer = torch.optim.SGD(tgt.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        cd.transfer_knowledge(
            data,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=1)
        cd.evaluate(data, criterion=criterion)

    def test_cross_lingual_translate(self):
        ml = load("cross_lingual.py")

        class DummyMultiLingual:
            def translate(self, text, sl, tl):
                return f"[{sl}->{tl}] {text}"

            def train(self):
                pass

            def eval(self):
                pass

            def __call__(self, x):
                return torch.zeros((len(x),))

        cl = ml.CROSS_LINGUAL(DummyMultiLingual())
        out = cl.translate("hello", "en", "fr")
        self.assertIn("[en->fr] hello", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
