"""`corerec train` / `corerec serve`: interactions file -> report -> artifact -> HTTP.

The demo in the README is `corerec serve sample_data/events.csv`. These tests
run that path on a small generated file, through the same functions the CLI
calls, and through the CLI entry point itself.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from corerec.serving.from_csv import (
    build_server,
    detect_columns,
    holdout_split,
    is_artifact,
    load_artifact,
    parse_params,
    read_interactions,
    save_artifact,
    train_from_csv,
)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _events(n_users=120, n_items=60, n_groups=4, seed=0):
    """Users read mostly from one group of items, in random order over time."""
    rng = np.random.default_rng(seed)
    item_group = rng.integers(0, n_groups, n_items)
    rows = []
    for u in range(n_users):
        group = u % n_groups
        own = np.flatnonzero(item_group == group)
        picks = list(rng.choice(own, size=min(8, len(own)), replace=False))
        picks += list(rng.choice(n_items, size=2, replace=False))
        rng.shuffle(picks)
        for t, i in enumerate(picks):
            rows.append((f"user{u}", f"item{i}", 5, 1_700_000_000 + 3600 * t + u))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def events_csv(tmp_path):
    path = tmp_path / "events.csv"
    _events().to_csv(path, index=False)
    return path


# -- column detection ---------------------------------------------------------

@pytest.mark.parametrize("columns,expected", [
    (["user_id", "item_id"], ("user_id", "item_id", None, None)),
    (["userId", "movieId", "rating", "timestamp"], ("userId", "movieId", "rating", "timestamp")),
    (["Customer ID", "SKU", "Quantity", "created_at"], ("Customer ID", "SKU", "Quantity", "created_at")),
    (["visitor", "product", "clicks"], ("visitor", "product", "clicks", None)),
])
def test_detect_columns(columns, expected):
    found = detect_columns(columns)
    assert (found["user"], found["item"], found["rating"], found["timestamp"]) == expected


def test_detect_columns_names_the_flag_when_it_cannot_guess():
    with pytest.raises(ValueError, match="--item-col"):
        detect_columns(["user_id", "thing"])


def test_explicit_column_must_exist():
    with pytest.raises(ValueError, match="'nope' is not in the file"):
        detect_columns(["a", "b"], user="nope", item="b")


def test_read_merges_duplicate_pairs(tmp_path):
    path = tmp_path / "dupes.csv"
    pd.DataFrame({"user": ["a", "a", "b"], "item": ["x", "x", "x"]}).to_csv(path, index=False)
    df, cols = read_interactions(path)
    assert cols["rating"] is None
    assert len(df) == 2
    assert df.set_index(["user", "item"]).loc[("a", "x"), "rating"] == 2.0


def test_read_parses_date_strings(tmp_path):
    path = tmp_path / "dated.csv"
    pd.DataFrame({"user_id": ["a", "a"], "item_id": ["x", "y"],
                  "date": ["2024-01-01", "2024-01-02"]}).to_csv(path, index=False)
    df, _ = read_interactions(path)
    assert df.sort_values("item")["timestamp"].tolist() == [1704067200, 1704153600]


# -- holdout ----------------------------------------------------------------------

def test_holdout_takes_each_users_latest_interactions():
    df = pd.DataFrame({"user": ["a"] * 5 + ["b"],
                       "item": list("vwxyz") + ["v"],
                       "rating": 1.0,
                       "timestamp": [5, 1, 4, 2, 3, 9]})
    train, test = holdout_split(df, test_fraction=0.4)
    assert sorted(test["item"]) == ["v", "x"]  # a's two latest; b has one row, stays in train
    assert set(train["user"]) == {"a", "b"}
    assert len(train) + len(test) == len(df)


# -- train, save, load, serve -----------------------------------------------------

def test_train_reports_a_model_that_beats_popularity(events_csv):
    result = train_from_csv(events_csv, model="ItemKNN")
    assert result.columns["user"] == "userId" and result.columns["item"] == "movieId"
    assert result.metrics["NDCG@10"] > result.baseline["NDCG@10"]
    report = result.report()
    assert "most popular" in report and "Warning" not in report


def test_report_warns_when_the_model_loses_to_popularity(events_csv):
    result = train_from_csv(events_csv, model="ItemKNN")
    result.metrics = dict(result.baseline)
    assert "does not beat recommending the most popular items" in result.report()


def test_unknown_and_content_models_are_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        train_from_csv(tmp_path / "missing.csv", model="NotAModel")
    with pytest.raises(ValueError, match="item text"):
        train_from_csv(tmp_path / "missing.csv", model="TFIDFRecommender")


@pytest.mark.parametrize("model,params", [("ALS", {"iterations": 3}), ("SAR", {}),
                                          ("EASE", {}), ("LightGCN", {"epochs": 2}),
                                          ("HSTU", {"epochs": 2, "embedding_dim": 16})])
def test_artifact_roundtrip_serves_the_same_recommendations(events_csv, tmp_path, model, params):
    result = train_from_csv(events_csv, model=model, params=params, evaluate=False)
    out = save_artifact(result, tmp_path / "artifact")
    assert is_artifact(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"] == model and manifest["popular_items"]

    loaded, manifest = load_artifact(out)
    assert loaded.recommend("user0", top_k=5) == result.model.recommend("user0", top_k=5)


def test_sequential_models_are_given_the_timestamps(events_csv):
    result = train_from_csv(events_csv, model="HSTU", params={"epochs": 1, "embedding_dim": 16},
                            evaluate=False)
    assert result.model.has_time


def test_server_falls_back_to_popular_items_for_unknown_users(events_csv, tmp_path):
    result = train_from_csv(events_csv, model="ItemKNN")
    client = TestClient(build_server(result.model, result.manifest()).app)

    known = client.post("/recommend", json={"user_id": "user0", "top_k": 5}).json()
    assert known["source"] == "model" and len(known["recommendations"]) == 5

    popular = result.manifest()["popular_items"]
    new = client.post("/recommend", json={"user_id": "nobody", "top_k": 3,
                                          "exclude_items": [popular[0]]}).json()
    assert new["source"] == "fallback"
    assert new["recommendations"] == popular[1:4]

    info = client.get("/info").json()
    assert info["artifact"]["metrics"]["NDCG@10"] == pytest.approx(result.metrics["NDCG@10"])


def test_fallback_also_covers_models_that_raise_for_unknown_users(events_csv):
    # SAR raises RecommendationError for an unseen user instead of returning [].
    result = train_from_csv(events_csv, model="SAR", evaluate=False)
    client = TestClient(build_server(result.model, result.manifest()).app)
    resp = client.post("/recommend", json={"user_id": "nobody", "top_k": 2})
    assert resp.status_code == 200 and resp.json()["source"] == "fallback"


def test_parse_params():
    assert parse_params(["factors=64", "reg=0.5", "name=x", "dims=[8, 4]"]) == {
        "factors": 64, "reg": 0.5, "name": "x", "dims": [8, 4]}
    with pytest.raises(ValueError):
        parse_params(["factors"])


# -- the CLI itself --------------------------------------------------------------

def test_cli_train_writes_a_servable_artifact(events_csv, tmp_path, monkeypatch, capsys):
    from corerec import cli

    out = tmp_path / "art"
    monkeypatch.setattr(sys, "argv", ["corerec", "train", str(events_csv), "-o", str(out),
                                      "--model", "ALS", "--param", "iterations=3"])
    assert cli.main() == 0
    printed = capsys.readouterr().out
    assert "NDCG@10" in printed and "most popular" in printed and str(out) in printed
    assert is_artifact(out)

    started = {}
    monkeypatch.setattr("corerec.serving.model_server.ModelServer.start",
                        lambda self, reload=False: started.setdefault("server", self))
    monkeypatch.setattr(sys, "argv", ["corerec", "serve", str(out), "--port", "9999"])
    assert cli.main() == 0
    server = started["server"]
    assert server.port == 9999 and server.metadata["model"] == "ALS"
