"""IdIndex, the shared id mapping for #76."""

import json

import numpy as np
import pytest

from corerec.api.id_index import IdIndex


@pytest.mark.parametrize("offset", [0, 1])
def test_codes_follow_first_appearance_and_offset(offset):
    idx, codes = IdIndex.fit(["b", "a", "b", 7, "a"], offset=offset)
    assert idx.ids == ["b", "a", 7]
    assert codes.tolist() == [offset, offset + 1, offset, offset + 2, offset + 1]
    assert idx.as_dict() == {"b": offset, "a": offset + 1, 7: offset + 2}
    assert [idx.id(c) for c in codes] == ["b", "a", "b", 7, "a"]
    assert len(idx) == 3 and "a" in idx and "z" not in idx


def test_unknown_id_raises_with_its_name():
    idx, _ = IdIndex.fit([1, 2])
    with pytest.raises(KeyError, match="3"):
        idx.codes([1, 3])


def test_json_roundtrip_keeps_numpy_ids_as_plain_values():
    idx, _ = IdIndex.fit(np.array([10, 20, 10]), offset=1)
    data = json.loads(json.dumps(idx.to_json()))  # what a safe bundle stores
    back = IdIndex.from_json(data)
    assert back.as_dict() == {10: 1, 20: 2}
    assert back.codes([20, 10]).tolist() == [2, 1]


def test_matches_the_mapping_models_build_today():
    """Same codes as corerec.nn.Recommender's pd.factorize + 1 (0 = padding)."""
    import pandas as pd

    items = ["x", "y", "x", "z"]
    codes, uniques = pd.factorize(pd.Series(items, dtype=object))
    idx, got = IdIndex.fit(items, offset=1)
    assert got.tolist() == (codes + 1).tolist()
    assert idx.ids == list(uniques)


def test_lightgcn_builds_its_maps_from_id_index(tmp_path):
    """Stage 2 of #76: LightGCN keeps its sorted codes, now via IdIndex."""
    from corerec.engines import LightGCN

    m = LightGCN(n_factors=4, n_layers=1, epochs=1, verbose=False, device="cpu").fit(
        [3, 1, 2, 1], ["b", "a", "c", "b"])
    assert m.user_id_map == m.users_index.as_dict() == {1: 0, 2: 1, 3: 2}
    assert m.item_id_map == {"a": 0, "b": 1, "c": 2}
    assert m.reverse_item_map == {0: "a", 1: "b", 2: "c"}
    m.save(tmp_path / "g")
    back = LightGCN.load(tmp_path / "g")
    assert back.user_id_map == m.user_id_map and back.recommend(1, top_k=2) == m.recommend(1, top_k=2)
