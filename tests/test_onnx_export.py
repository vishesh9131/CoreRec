"""ONNX export serves the same ranking the Python model does."""

import json

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import corerec.engines as E  # noqa: E402
from corerec.export import to_onnx  # noqa: E402

KW = {
    "TwoTower": dict(embedding_dim=16, epochs=2, verbose=False),
    "DCN": dict(embedding_dim=8, epochs=2),
    "DeepFM": dict(embedding_dim=8, epochs=2),
    "SASRec": dict(hidden_units=16, num_blocks=1, epochs=2, max_seq_length=20, verbose=False),
    "HSTU": dict(embedding_dim=16, num_heads=2, num_blocks=1, epochs=2, max_seq_length=20),
    "MultVAE": dict(hidden_dim=32, latent_dim=8, epochs=3),
    "MultiDAE": dict(hidden_dim=32, latent_dim=8, epochs=3),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    # non-contiguous raw ids, so index/id mix-ups show up
    return (rng.integers(0, 40, 600) * 3 + 7).tolist(), (rng.integers(0, 80, 600) * 5 + 11).tolist()


@pytest.mark.parametrize("name", list(KW))
def test_onnx_ranks_like_the_model(name, data, tmp_path):
    u, i = data
    m = getattr(E, name)(device="cpu", **KW[name])
    m.fit(u, i, [1.0] * len(u))
    sess = ort.InferenceSession(str(to_onnx(m, tmp_path / "m.onnx")))
    meta = sess.get_modelmeta().custom_metadata_map
    assert meta["corerec_model"] == name
    items = json.loads(meta["item_ids"])
    inp = sess.get_inputs()[0].name

    for user in sorted(set(u))[:10]:
        if inp == "history":
            # the graph's width, which for HSTU is the fitted window
            n = int(meta["max_seq_length"])
            seq = m.user_sequences[user][-n:]
            x = np.zeros((1, n), np.int64)
            x[0, -len(seq):] = seq
        elif inp == "interactions":
            x = m.R[m.user_map[user]].toarray().astype(np.float32)
        else:
            x = np.array([json.loads(meta["user_ids"]).index(user)], np.int64)
        scores = sess.run(None, {inp: x})[0][0]
        assert len(scores) == len(items)
        if name in ("DCN", "DeepFM", "MultVAE", "MultiDAE"):
            # VAE recommend() always drops seen items, so compare raw scores
            np.testing.assert_allclose(scores, m._score_all_items(user), atol=1e-4)
        else:
            top = [items[j] for j in np.argsort(-scores)[:5]]
            assert top == m.recommend(user, top_k=5, exclude_seen=False)


@pytest.mark.parametrize("name", list(KW))
def test_onnx_batch_dimension_is_dynamic(name, data, tmp_path):
    u, i = data
    m = getattr(E, name)(device="cpu", **KW[name])
    m.fit(u, i, [1.0] * len(u))
    sess = ort.InferenceSession(str(to_onnx(m, tmp_path / "m.onnx")))
    meta = sess.get_modelmeta().custom_metadata_map
    inp = sess.get_inputs()[0].name
    if inp == "user_index":
        x = np.arange(7, dtype=np.int64)
    elif inp == "interactions":
        x = m.R[:7].toarray().astype(np.float32)
    else:
        n = int(meta["max_seq_length"])
        x = np.tile(np.arange(1, n + 1, dtype=np.int64), (7, 1))
    one = sess.run(None, {inp: x[:1]})[0]
    seven = sess.run(None, {inp: x})[0]
    assert seven.shape[0] == 7
    np.testing.assert_allclose(seven[:1], one, atol=1e-5)


def test_hstu_with_timestamps_takes_them_as_a_second_input(data, tmp_path):
    u, i = data
    t = (np.arange(len(u)) * 3600.0).tolist()  # an event an hour, so time buckets vary
    m = E.HSTU(device="cpu", **KW["HSTU"])
    m.fit(u, i, timestamps=t)
    assert m.has_time
    sess = ort.InferenceSession(str(to_onnx(m, tmp_path / "m.onnx")))
    assert [x.name for x in sess.get_inputs()] == ["history", "timestamps"]
    n = int(sess.get_modelmeta().custom_metadata_map["max_seq_length"])

    users = sorted(set(u))[:10]
    X = np.zeros((len(users), n), np.int64)
    T = np.zeros((len(users), n), np.float32)
    for r, user in enumerate(users):
        seq, ts = m.user_sequences[user][-n:], m.user_times[user][-n:]
        X[r, -len(seq):], T[r, -len(ts):] = seq, ts
    scores = sess.run(None, {"history": X, "timestamps": T})[0]
    np.testing.assert_allclose(scores, m.score_users(users), atol=1e-4)
    # rows don't leak into each other across the batch
    one = sess.run(None, {"history": X[3:4], "timestamps": T[3:4]})[0]
    np.testing.assert_allclose(one, scores[3:4], atol=1e-5)


def test_unsupported_model_says_what_to_use_instead(data, tmp_path):
    u, i = data
    m = E.ALS(epochs=2)
    m.fit(u, i)
    with pytest.raises(NotImplementedError, match="ModelServer"):
        to_onnx(m, tmp_path / "m.onnx")
