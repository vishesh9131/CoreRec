import pytest

from corerec.engines import SAR


@pytest.mark.parametrize("persist", [False, True])
def test_sar_can_include_seen_items(tmp_path, persist):
    model = SAR(similarity_type="jaccard").fit([1, 1, 2, 2], [10, 11, 11, 12], [1.] * 4)
    if persist:
        model.save(tmp_path / "sar")
        model = SAR.load(tmp_path / "sar")
    assert model.recommend(1, top_k=3) == [12]
    assert set(model.recommend(1, top_k=3, exclude_seen=False)) == {10, 11, 12}
    assert 10 not in model.recommend(1, top_k=3, exclude_seen=False, exclude_items=[10])
