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
            seq = m.user_sequences[user][-m.max_seq_length:]
            x = np.zeros((1, m.max_seq_length), np.int64)
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
    inp = sess.get_inputs()[0].name
    if inp == "user_index":
        x = np.arange(7, dtype=np.int64)
    elif inp == "interactions":
        x = m.R[:7].toarray().astype(np.float32)
    else:
        x = np.tile(np.arange(1, m.max_seq_length + 1, dtype=np.int64), (7, 1))
    one = sess.run(None, {inp: x[:1]})[0]
    seven = sess.run(None, {inp: x})[0]
    assert seven.shape[0] == 7
    np.testing.assert_allclose(seven[:1], one, atol=1e-5)


def test_unsupported_model_says_what_to_use_instead(data, tmp_path):
    u, i = data
    m = E.ALS(epochs=2)
    m.fit(u, i)
    with pytest.raises(NotImplementedError, match="ModelServer"):
        to_onnx(m, tmp_path / "m.onnx")
