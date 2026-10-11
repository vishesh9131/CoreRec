"""verbose=True shows training progress without the app configuring logging (#209).

Progress goes through logger.info, and Python drops INFO records when nothing
configured logging, so verbose=True printed nothing. pytest itself puts a
handler on the root logger, so these tests remove it to look like a plain
script or notebook.
"""

import contextlib
import logging

import numpy as np
import pytest

import corerec.engines as engines

rng = np.random.default_rng(0)
USERS, ITEMS = rng.integers(0, 30, 400).tolist(), rng.integers(0, 40, 400).tolist()
RATINGS = rng.integers(1, 6, 400).astype(float).tolist()

MODELS = [("DCN", {}), ("DeepFM", {}), ("LightGCN", {}), ("TwoTower", {})]


@contextlib.contextmanager
def plain_script():
    # pytest adds its capture handler to root for each test phase, so this has
    # to run inside the test body, not in a fixture
    root, lg = logging.getLogger(), logging.getLogger("corerec")
    saved, before = root.handlers, (list(lg.handlers), lg.level)
    root.handlers = []
    try:
        yield
    finally:
        root.handlers = saved
    assert (list(lg.handlers), lg.level) == before  # nothing left behind after fit


@pytest.mark.parametrize("name,kw", MODELS, ids=[m[0] for m in MODELS])
def test_verbose_prints_progress(capsys, name, kw):
    with plain_script():
        getattr(engines, name)(verbose=True, epochs=2, device="cpu", **kw).fit(USERS, ITEMS, RATINGS)
    err = capsys.readouterr().err
    assert "2/2" in err  # the last epoch's line


@pytest.mark.parametrize("name,kw", MODELS, ids=[m[0] for m in MODELS])
def test_quiet_stays_quiet(capsys, name, kw):
    with plain_script():
        getattr(engines, name)(verbose=False, epochs=2, device="cpu", **kw).fit(USERS, ITEMS, RATINGS)
    assert "2/2" not in capsys.readouterr().err


def test_an_app_that_configured_logging_keeps_control(capsys):
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    with plain_script():
        logging.getLogger().handlers = [handler]
        engines.DCN(verbose=True, epochs=1, device="cpu").fit(USERS, ITEMS, RATINGS)
    assert "epoch" not in capsys.readouterr().err  # their WARNING handler decides
