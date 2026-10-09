"""corerec retrain: champion/challenger on data neither model has seen."""

import json
from pathlib import Path

import pandas as pd
import pytest

from corerec.serving.from_csv import load_artifact, retrain_artifact, save_artifact, train_from_csv

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "events.csv"


@pytest.fixture()
def files(tmp_path):
    df = pd.read_csv(SAMPLE).sort_values("timestamp").head(12000)
    n = int(len(df) * 0.7)
    df.iloc[:n].to_csv(tmp_path / "old.csv", index=False)
    df.to_csv(tmp_path / "now.csv", index=False)
    art = tmp_path / "art"
    save_artifact(train_from_csv(tmp_path / "old.csv", model="ItemKNN", evaluate=False), art)
    return tmp_path, art, df


def test_nothing_new_keeps_the_model(files):
    tmp, art, _ = files
    d = retrain_artifact(art)
    assert not d["promoted"] and d["new_rows"] == 0 and "new rows" in d["reason"]
    assert not (art / "previous").exists()


def test_new_rows_are_backtested_and_promoted(files):
    tmp, art, df = files
    before = json.loads((art / "manifest.json").read_text())["trained_through"]
    d = retrain_artifact(art, data=tmp / "now.csv")
    assert d["new_rows"] == (df["timestamp"] > before).sum()
    assert d["current"] > 0, "the deployed model must be scored on rows it hasn't seen"
    # fresher data wins on this fixture (0.130 vs 0.111 NDCG@10)
    assert d["promoted"] and d["candidate"] >= d["current"]
    assert (art / "previous" / "manifest.json").exists()
    model, manifest = load_artifact(art)
    assert manifest["trained_through"] == df["timestamp"].max()
    assert manifest["retrain"]["candidate"] == d["candidate"]


def test_a_worse_candidate_is_not_shipped(files):
    tmp, art, _ = files
    original = (art / "manifest.json").read_text()
    d = retrain_artifact(art, data=tmp / "now.csv", tolerance=-1.0)  # demand +1.0 NDCG
    assert d["would_promote"] is False and d["promoted"] is False
    assert (art / "manifest.json").read_text() == original


def test_dry_run_changes_nothing(files):
    tmp, art, _ = files
    original = (art / "manifest.json").read_text()
    d = retrain_artifact(art, data=tmp / "now.csv", dry_run=True)
    assert d["promoted"] is False and d["candidate"] is not None
    assert (art / "manifest.json").read_text() == original


def test_feedback_clicks_count_as_new_data(files):
    from corerec.serving.feedback import FeedbackLog

    tmp, art, df = files
    log = FeedbackLog(tmp / "fb.jsonl")
    for u, i in df[["user_id", "item_id"]].head(300).itertuples(index=False):
        log.feedback(u, i, request_id=log.impression(u, [i]))
    d = retrain_artifact(art, feedback=tmp / "fb.jsonl", dry_run=True)
    assert d["feedback_rows"] == 300 and d["new_rows"] == 300


def test_append_only_file_without_timestamps_is_retrained(tmp_path):
    """No time column: rows appended since the last training are the new data (#48)."""
    df = pd.read_csv(SAMPLE).sort_values("timestamp").head(12000).drop(columns="timestamp")
    n = int(len(df) * 0.7)
    log = tmp_path / "log.csv"
    df.iloc[:n].to_csv(log, index=False)
    art = tmp_path / "a"
    save_artifact(train_from_csv(log, model="ItemKNN", evaluate=False), art)
    assert json.loads((art / "manifest.json").read_text())["trained_rows"] == n

    assert retrain_artifact(art)["new_rows"] == 0  # same file, nothing new

    df.to_csv(log, index=False)  # the log grew
    d = retrain_artifact(art)
    assert d["mode"] == "row order"
    assert d["new_rows"] == len(df) - n
    assert d["current"] > 0, "the deployed model must be scored on rows it hasn't seen"
    if d["promoted"]:
        assert json.loads((art / "manifest.json").read_text())["trained_rows"] == len(df)


def test_rewritten_file_without_timestamps_is_refused(tmp_path):
    df = pd.read_csv(SAMPLE).drop(columns="timestamp").head(3000)
    df.to_csv(tmp_path / "e.csv", index=False)
    save_artifact(train_from_csv(tmp_path / "e.csv", model="ItemKNN", evaluate=False), tmp_path / "a")
    df.head(1000).to_csv(tmp_path / "e.csv", index=False)  # shrank: not append-only
    with pytest.raises(ValueError, match="append-only"):
        retrain_artifact(tmp_path / "a")


def test_feedback_counts_as_new_data_without_timestamps(tmp_path):
    from corerec.serving.feedback import FeedbackLog

    df = pd.read_csv(SAMPLE).drop(columns="timestamp").head(6000)
    df.to_csv(tmp_path / "e.csv", index=False)
    art = tmp_path / "a"
    save_artifact(train_from_csv(tmp_path / "e.csv", model="ItemKNN", evaluate=False), art)
    log = FeedbackLog(tmp_path / "fb.jsonl")
    for u, i in df.sample(300, random_state=0)[["user_id", "item_id"]].itertuples(index=False):
        log.feedback(u, i, request_id=log.impression(u, [i]))
    d = retrain_artifact(art, feedback=tmp_path / "fb.jsonl", min_new_rows=50, dry_run=True)
    assert d["mode"] == "row order" and d["feedback_rows"] == 300 and d["new_rows"] == 300


def test_served_artifact_reloads_after_retrain(files):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from corerec.serving.from_csv import build_server

    tmp, art, _ = files
    model, manifest = load_artifact(art)
    client = TestClient(build_server(model, manifest, artifact=art,
                                     feedback_log=tmp / "fb.jsonl").app)
    retrain_artifact(art, data=tmp / "now.csv")
    assert client.post("/reload").json()["status"] == "reloaded"
    body = client.post("/recommend", json={"user_id": "u0042", "top_k": 5}).json()
    assert len(body["recommendations"]) == 5 and body["request_id"]
