"""Run the quickstart example end to end.

examples/train_and_serve.py is the path the README promises: interactions ->
trained model -> HTTP endpoint returning recommendations. Running it here means
the example cannot silently break, which is how quickstarts normally rot.

It caught two real bugs when first written:
  - ModelServer's /recommend passes exclude_items, which TwoTower.recommend did
    not accept, so every request against a two-tower model returned HTTP 500.
  - TwoTower.recommend accepted exclude_seen and never read it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "train_and_serve.py"

pytest.importorskip("fastapi", reason="serving extra not installed")


def _load_example():
    spec = importlib.util.spec_from_file_location("train_and_serve", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_and_serve"] = module
    spec.loader.exec_module(module)
    return module


def test_example_file_exists():
    assert EXAMPLE.is_file(), f"README quickstart points at {EXAMPLE}, which is missing"


def test_train_and_serve_end_to_end():
    """Train, serve over HTTP, and get back a usable ranking."""
    recs = _load_example().main()
    assert len(recs) == 5
    assert len(set(recs)) == 5, f"duplicate recommendations: {recs}"


def test_exclude_items_is_honoured():
    """The /recommend contract lets callers drop specific items."""
    from fastapi.testclient import TestClient

    from corerec.serving import ModelServer

    example = _load_example()
    users, items, ratings, _, _ = example.build_interactions()
    model = example.TwoTower(embedding_dim=16, num_epochs=5, verbose=False)
    model.fit(users, items, ratings)

    client = TestClient(ModelServer(model).app)
    baseline = client.post("/recommend", json={"user_id": 0, "top_k": 5}).json()
    banned = baseline["recommendations"][:2]

    filtered = client.post(
        "/recommend", json={"user_id": 0, "top_k": 5, "exclude_items": banned}
    ).json()["recommendations"]

    assert not set(banned) & set(filtered), f"{banned} leaked into {filtered}"
    assert len(filtered) == 5, "excluding items should not shorten the list"
