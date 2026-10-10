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


def test_tokens_guard_reload_and_feedback(good, tmp_path):
    """Without a token anyone reaching the port could swap the model or write
    clicks that retrain learns from (#47)."""
    server = ModelServer(good, feedback_log=tmp_path / "fb.jsonl", reload_fn=lambda: Worst(),
                         admin_token="adm", feedback_token="fbk")
    client = TestClient(server.app)
    click = {"user_id": 0, "item_id": 1, "event": "click"}

    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"Authorization": "Bearer fbk"}).status_code == 401
    assert client.post("/feedback", json=click).status_code == 401
    assert client.post("/feedback", json=click, headers={"Authorization": "Bearer adm"}).status_code == 401
    from corerec.serving.feedback import FeedbackLog
    assert not any(r["type"] == "feedback" for r in FeedbackLog(tmp_path / "fb.jsonl").records())

    assert client.post("/feedback", json=click, headers={"Authorization": "Bearer fbk"}).status_code == 200
    r = client.post("/reload", headers={"Authorization": "Bearer adm"})
    assert r.status_code == 200 and r.json()["model"] == "Worst"
    # reading endpoints stay open
    assert client.post("/recommend", json={"user_id": 0, "top_k": 3}).status_code == 200


def test_no_tokens_keeps_endpoints_open(good, tmp_path):
    server = ModelServer(good, feedback_log=tmp_path / "fb.jsonl", reload_fn=lambda: Worst())
    client = TestClient(server.app)
    assert client.post("/feedback", json={"user_id": 0, "item_id": 1}).status_code == 200
    assert client.post("/reload").status_code == 200


def _write_log(path, n_requests, seed=0):
    """A feedback log written directly (log.impression() per event is slow at 100k)."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for n in range(n_requests):
            items = rng.integers(0, 3000, 10).tolist()
            f.write(json.dumps({"type": "impression", "request_id": f"r{n}", "ts": float(n),
                                "user_id": n % 5000, "items": items,
                                "variant": "a" if n % 2 else "b", "source": "model"}) + "\n")
            if n % 4 == 0:
                f.write(json.dumps({"type": "feedback", "request_id": f"r{n}", "ts": n + 0.5,
                                    "user_id": n % 5000, "item_id": items[n % 10],
                                    "event": "click"}) + "\n")


def test_metrics_on_a_100k_request_log(tmp_path):
    """/metrics re-read and re-joined the whole log on every call (#40)."""
    path = tmp_path / "fb.jsonl"
    _write_log(path, 100_000)
    log = FeedbackLog(path)
    m = log.metrics()["all"]
    assert m["requests"] == 100_000 and m["impressions"] == 1_000_000
    assert m["clicks"] >= 25_000  # one click per 4th request, more when the item repeats
    assert log._offset == path.stat().st_size

    log.feedback(1, 2, request_id=log.impression(1, [2, 3]))
    m2 = log.metrics()["all"]
    assert (m2["requests"], m2["clicks"]) == (100_001, m["clicks"] + 1)
    assert log.drift(recent=1000)["recent"]["requests"] == 1000


def test_metrics_follow_appends_partial_lines_and_rotation(tmp_path):
    path = tmp_path / "fb.jsonl"
    writer, reader = FeedbackLog(path), FeedbackLog(path)  # e.g. server and a monitoring job
    rid = writer.impression(1, ["x", "y"])
    assert reader.metrics()["all"]["clicks"] == 0

    line = json.dumps({"type": "feedback", "request_id": rid, "ts": 1.0, "user_id": 1,
                       "item_id": "y", "event": "click"}) + "\n"
    with open(path, "a") as f:
        f.write(line[:20])  # writer caught mid-line
    assert reader.metrics()["all"]["clicks"] == 0
    with open(path, "a") as f:
        f.write(line[20:])
    m = reader.metrics()["all"]
    assert m["clicks"] == 1 and m["mrr"] == 0.5

    path.write_text("")  # rotated
    writer.impression(2, ["z"])
    assert reader.metrics()["all"]["requests"] == 1
