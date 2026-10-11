"""user_guide/custom_models.md runs top to bottom (#224).

The page's serving block loaded the saved custom model through ModelLoader,
which couldn't pass module_cls, and called a server.run() that doesn't exist.
Every python block runs in order in a temp directory; server.start() is
replaced so nothing binds a port.
"""

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "user_guide" / "custom_models.md"
BLOCKS = re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S)


def test_custom_models_page_runs(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from corerec.serving import ModelServer

    started = []
    monkeypatch.setattr(ModelServer, "start", lambda self, reload=False: started.append(self))
    monkeypatch.chdir(tmp_path)
    ns = {"__name__": "custom_models_page"}
    for i, block in enumerate(BLOCKS):
        if "to_onnx" in block and not _has("onnx"):
            continue
        code = block.replace("server = ModelServer(same)", "server = ModelServer(same); server.start()")
        try:
            exec(compile(code, f"{PAGE.name}[block {i}]", "exec"), ns)
        except Exception as e:  # name the block so a doc break is easy to find
            pytest.fail(f"block {i} of {PAGE.name} failed: {type(e).__name__}: {e}\n{block}")
    assert started and started[0].model.recommend(0, top_k=5) == ns["rec"].recommend(0, top_k=5)


def _has(module):
    import importlib.util

    return importlib.util.find_spec(module) is not None


def test_model_loader_takes_module_cls(tmp_path):
    import torch.nn as nn

    from corerec.nn import Recommender
    from corerec.serving import ModelLoader

    class Dot(nn.Module):
        def __init__(self, n_users, n_items, dim=8):
            super().__init__()
            self.u, self.i = nn.Embedding(n_users, dim), nn.Embedding(n_items + 1, dim, padding_idx=0)

        def forward(self, users, items):
            return (self.u(users).unsqueeze(1) * self.i(items)).sum(-1)

    rec = Recommender(Dot, {"dim": 8}, loss="bpr", epochs=1).fit([0, 0, 1, 1, 2], [1, 2, 2, 3, 1])
    rec.save(tmp_path / "dot.pt")
    with pytest.raises(ValueError, match="module_cls"):
        ModelLoader().load(tmp_path / "dot.pt")
    back = ModelLoader().load(tmp_path / "dot.pt", module_cls=Dot)
    assert back.recommend(0, top_k=3) == rec.recommend(0, top_k=3)


def test_cli_serves_a_custom_model_with_module(tmp_path, monkeypatch):
    import sys

    pytest.importorskip("fastapi")
    from corerec import cli
    from corerec.nn import Recommender
    from corerec.serving import ModelServer

    (tmp_path / "mymodels.py").write_text(
        "import torch.nn as nn\n\n"
        "class Dot(nn.Module):\n"
        "    def __init__(self, n_users, n_items, dim=8):\n"
        "        super().__init__()\n"
        "        self.u, self.i = nn.Embedding(n_users, dim), nn.Embedding(n_items + 1, dim, padding_idx=0)\n\n"
        "    def forward(self, users, items):\n"
        "        return (self.u(users).unsqueeze(1) * self.i(items)).sum(-1)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import mymodels

    rec = Recommender(mymodels.Dot, {"dim": 8}, loss="bpr", epochs=1).fit([0, 0, 1, 1, 2], [1, 2, 2, 3, 1])
    rec.save(tmp_path / "dot.pt")

    started = []
    monkeypatch.setattr(ModelServer, "start", lambda self, reload=False: started.append(self))
    monkeypatch.setattr(sys, "argv", ["corerec", "serve", str(tmp_path / "dot.pt"), "--module", "mymodels:Dot"])
    assert cli.main() == 0
    assert started[0].model.recommend(0, top_k=3) == rec.recommend(0, top_k=3)

    monkeypatch.setattr(sys, "argv", ["corerec", "serve", str(tmp_path / "dot.pt"), "--module", "mymodels"])
    with pytest.raises(SystemExit, match="package.module:ClassName"):
        cli.main()
