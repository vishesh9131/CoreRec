"""towers and integrations moved under corerec.experimental (#79); old paths still work."""

import importlib
import sys

import pytest


def test_integrations_new_path_and_old_path_warns():
    new = importlib.import_module("corerec.experimental.integrations")
    sys.modules.pop("corerec.integrations", None)
    with pytest.warns(DeprecationWarning, match="corerec.experimental.integrations"):
        old = importlib.import_module("corerec.integrations")
    assert old.MLflowTracker is new.MLflowTracker and old.WandBTracker is new.WandBTracker


def test_towers_new_path_and_old_path_warns():
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    new = importlib.import_module("corerec.experimental.towers")
    sys.modules.pop("corerec.towers", None)
    with pytest.warns(DeprecationWarning, match="corerec.experimental.towers"):
        old = importlib.import_module("corerec.towers")
    assert old.MLPTower is new.MLPTower
