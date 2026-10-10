import numpy as np
import pytest

from corerec.engines import DCN, DeepFM, SAR


def test_deepfm_keeps_unseen_batch_candidates():
    model = DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, device="cpu")
    model.fit([1, 1, 2, 2], [10, 11, 11, 12], [1.] * 4)
    assert model.recommend(1, top_k=3) == [12]


@pytest.mark.parametrize("cls,kwargs", [(DCN, {"deep_layers": [8]}),
                                      (DeepFM, {"hidden_layers": [8]}), (SAR, {})])
def test_scored_recommendations_match_ranking_and_prediction(cls, kwargs, tmp_path):
    if cls is not SAR:
        kwargs = {**kwargs, "embedding_dim": 4, "epochs": 1, "device": "cpu"}
    model = cls(**kwargs).fit([1, 1, 2, 2], [10, 11, 11, 12], [1.] * 4)
    path = tmp_path / "model"
    model.save(path)
    for candidate in (model, cls.load(path)):
        ids = candidate.recommend(1, top_k=3)
        scored = candidate.recommend(1, top_k=3, return_scores=True)
        assert [item for item, score in scored] == ids == [12]
        for item, score in scored:
            assert isinstance(score, float)
            assert np.isfinite(score)
            assert score == pytest.approx(candidate.predict(1, item), rel=1e-5, abs=1e-6)
        assert candidate.recommend("unknown", top_k=3, return_scores=True) == []
        assert candidate.recommend(1, top_k=3, exclude_items=[12], return_scores=True) == []


def test_deepfm_legacy_checkpoint_restores_history(tmp_path):
    model = DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, device="cpu")
    model.fit(["01", "01", "02", "02"], [10, 11, 11, 12], [1.] * 4)
    path = tmp_path / "legacy"
    model.save(path, safe=False)
    with pytest.warns(UserWarning):
        restored = DeepFM.load(path, allow_pickle=True)
    assert restored.recommend("01", top_k=3) == [12]
    assert set(restored.recommend("01", top_k=3, exclude_seen=False)) == {10, 11, 12}


def test_deepfm_interleaved_exclusions_keep_scores_aligned():
    model = DeepFM(embedding_dim=4, hidden_layers=[8], epochs=1, device="cpu")
    model.fit([1, 1, 2, 2, 2], [10, 12, 11, 13, 14], [1.] * 5)
    scored = model.recommend(1, top_k=5, exclude_seen=False,
                             exclude_items=[10, 12], return_scores=True)
    assert {item for item, score in scored} == {11, 13, 14}
    for item, score in scored:
        assert score == pytest.approx(model.predict(1, item), rel=1e-5, abs=1e-6)
