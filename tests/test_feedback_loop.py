"""Feedback logging, online metrics, A/B tests and drift alerts on ModelServer."""

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from corerec.serving import ModelServer  # noqa: E402
from corerec.serving.feedback import FeedbackLog, assign_variant  # noqa: E402

N_GROUPS, PER_GROUP = 4, 25


def _group(item):
    return item // PER_GROUP


def _data(n_users=200, seed=0):
    rng = np.random.default_rng(seed)
    users, items = [], []
    for u in range(n_users):
        g = u % N_GROUPS
        for it in rng.choice(np.arange(g * PER_GROUP, (g + 1) * PER_GROUP), 8, replace=False):
            users.append(u)
            items.append(int(it))
    return users, items


class Worst:
    """Recommends from the wrong group: the "bad" arm of the A/B test."""

    def recommend(self, user_id, top_k=10, exclude_items=None, **_):
        g = (user_id % N_GROUPS + 1) % N_GROUPS
        return list(range(g * PER_GROUP, g * PER_GROUP + top_k))

    def predict(self, user_id, item_id, **_):
        return 0.0


@pytest.fixture(scope="module")
def good():
    from corerec.engines import ItemKNN

    return ItemKNN().fit(*_data())


def _click_through(client, users, rng, p_click=0.3):
    """Simulated users click shown items from their own group with p_click."""
    for u in users:
        body = client.post("/recommend", json={"user_id": u, "top_k": 5}).json()
        for item in body["recommendations"]:
            if _group(item) == u % N_GROUPS and rng.random() < p_click:
                client.post("/feedback", json={"user_id": u, "item_id": item,
                                               "request_id": body["request_id"]})


def test_impressions_and_clicks_become_metrics(good, tmp_path):
    client = TestClient(ModelServer(good, feedback_log=tmp_path / "fb.jsonl").app)
    body = client.post("/recommend", json={"user_id": 0, "top_k": 5}).json()
    assert body["variant"] == "default" and body["request_id"]
    client.post("/feedback", json={"user_id": 0, "item_id": body["recommendations"][2],
                                   "request_id": body["request_id"]})
    m = client.get("/metrics").json()["variants"]["all"]
    assert m["requests"] == 1 and m["impressions"] == 5 and m["clicks"] == 1
    assert m["ctr"] == pytest.approx(0.2)
    assert m["mrr"] == pytest.approx(1 / 3)


def test_metrics_with_no_traffic_are_null_not_nan(good, tmp_path):
    client = TestClient(ModelServer(good, feedback_log=tmp_path / "fb.jsonl").app)
    assert client.get("/metrics").json()["variants"]["all"]["ctr"] is None


def test_feedback_endpoints_are_off_without_a_log(good):
    client = TestClient(ModelServer(good).app)
    assert client.post("/feedback", json={"user_id": 0, "item_id": 1}).status_code == 404
    assert "request_id" not in client.post("/recommend", json={"user_id": 0, "top_k": 3}).json()


def test_assignment_is_sticky_and_follows_traffic():
    traffic = {"control": 0.8, "treatment": 0.2}
    assert all(assign_variant(u, traffic) == assign_variant(u, traffic) for u in range(100))
    share = np.mean([assign_variant(u, traffic) == "treatment" for u in range(20000)])
    assert abs(share - 0.2) < 0.02


def test_ab_test_finds_the_better_model(good, tmp_path):
    server = ModelServer({"control": Worst(), "treatment": good},
                         traffic={"control": 0.5, "treatment": 0.5},
                         feedback_log=tmp_path / "fb.jsonl")
    client = TestClient(server.app)
    _click_through(client, range(200), np.random.default_rng(0))
    out = client.get("/metrics").json()
    ab = out["ab_test"]
    assert out["variants"]["treatment"]["ctr"] > out["variants"]["control"]["ctr"]
    assert ab["significant_at_0.05"]
    assert ab["ctr_treatment"] > ab["ctr_control"]


def test_mismatched_traffic_is_rejected(good):
    with pytest.raises(ValueError, match="traffic"):
        ModelServer({"a": good, "b": good}, traffic={"a": 1.0})


def _synthetic_log(path, n, fallback_share, click_items, ctr, rng, start):
    log = FeedbackLog(path)
    for k in range(n):
        source = "fallback" if rng.random() < fallback_share else "model"
        rid = log.impression(k, list(range(5)), source=source)
        if rng.random() < ctr * 5:
            log.feedback(k, int(rng.choice(click_items)), request_id=rid)
    return log


def test_drift_is_quiet_on_stable_traffic(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "fb.jsonl"
    _synthetic_log(path, 400, 0.05, [0, 1, 2, 3, 4], 0.1, rng, 0)
    log = _synthetic_log(path, 200, 0.05, [0, 1, 2, 3, 4], 0.1, rng, 400)
    assert log.drift(recent=200)["alerts"] == []


def test_drift_alerts_on_fallback_and_ctr(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "fb.jsonl"
    _synthetic_log(path, 400, 0.05, [0, 1, 2, 3, 4], 0.1, rng, 0)
    log = _synthetic_log(path, 200, 0.5, [0, 1, 2, 3, 4], 0.02, rng, 400)
    alerts = " ".join(log.drift(recent=200)["alerts"])
    assert "fallback" in alerts and "CTR" in alerts


def test_clicks_export_for_retraining(tmp_path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    rid = log.impression("u1", ["a", "b"])
    log.feedback("u1", "b", request_id=rid)
    log.feedback("u1", "a", event="view")
    events = log.to_events()
    assert events[["user_id", "item_id"]].values.tolist() == [["u1", "b"]]


def test_reload_swaps_the_model(good, tmp_path):
    server = ModelServer(good, reload_fn=lambda: Worst())
    client = TestClient(server.app)
    assert client.post("/reload").json()["model"] == "Worst"
    recs = client.post("/recommend", json={"user_id": 0, "top_k": 3}).json()["recommendations"]
    assert recs == Worst().recommend(0, top_k=3)


def test_log_lines_are_plain_json(tmp_path):
    log = FeedbackLog(tmp_path / "fb.jsonl")
    log.impression(np.int64(3), [np.int64(1), 2])
    rec = json.loads((tmp_path / "fb.jsonl").read_text().splitlines()[0])
    assert rec["user_id"] == 3 and rec["items"] == [1, 2]
