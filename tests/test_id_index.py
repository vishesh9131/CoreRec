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
    """LightGCN uses IdIndex first-observed codes and retains them on load."""
    from corerec.engines import LightGCN

    m = LightGCN(n_factors=4, n_layers=1, epochs=1, verbose=False, device="cpu").fit(
        [3, 1, 2, 1], ["b", "a", "c", "b"])
    assert m.user_id_map == m.users_index.as_dict() == {3: 0, 1: 1, 2: 2}
    assert m.item_id_map == {"b": 0, "a": 1, "c": 2}
    assert m.reverse_item_map == {0: "b", 1: "a", 2: "c"}
    m.save(tmp_path / "g")
    back = LightGCN.load(tmp_path / "g")
    assert back.user_id_map == m.user_id_map and back.recommend(1, top_k=2) == m.recommend(1, top_k=2)


def test_nn_recommender_builds_its_maps_from_id_index(tmp_path):
    """Stage 2 of #76, first model: same public maps, one source for them."""
    from corerec.nn import MatrixFactorization, Recommender

    users, items = ["u1", "u2", "u1", "u3"], [10, 20, 30, 10]
    rec = Recommender(MatrixFactorization, {"dim": 4}, epochs=1, device="cpu").fit(users, items)
    assert rec.user_map == rec.users_index.as_dict() == {"u1": 0, "u2": 1, "u3": 2}
    assert rec.item_map == rec.items_index.as_dict() == {10: 1, 20: 2, 30: 3}  # 0 = padding
    rec.save(tmp_path / "m.pt")
    back = Recommender.load(tmp_path / "m.pt", device="cpu")
    assert back.user_map == rec.user_map and back.item_map == rec.item_map
    assert back.recommend("u1", top_k=2) == rec.recommend("u1", top_k=2)


def test_two_tower_builds_its_maps_from_id_index():
    """Stage 2 of #76: TwoTower's maps come from IdIndex; public dicts unchanged."""
    from corerec.engines import TwoTower

    m = TwoTower(embedding_dim=4, epochs=1, verbose=False, device="cpu").fit(
        ["u2", "u1", "u2"], [30, 10, 20])
    assert m.user_map == m.users_index.as_dict()
    assert m.item_map == m.items_index.as_dict()
    assert m.reverse_item_map == {v: k for k, v in m.item_map.items()}


def test_hstu_builds_its_item_map_from_id_index(tmp_path):
    """Stage 2 of #76: HSTU's item map comes from IdIndex; public attributes unchanged."""
    from corerec.engines import HSTU

    users, items = ["a", "a", "a", "b", "b"], ["x", "y", "x", "z", "y"]
    m = HSTU(embedding_dim=8, num_heads=1, num_blocks=1, epochs=1, device="cpu").fit(users, items)
    assert m.item_to_index == m.items_index.as_dict() == {"x": 1, "y": 2, "z": 3}
    assert m.index_to_item == [None, "x", "y", "z"]
    m.save(tmp_path / "h")
    back = HSTU.load(tmp_path / "h")
    assert back.item_to_index == m.item_to_index and back.index_to_item == m.index_to_item
    assert back.recommend("a", top_k=2) == m.recommend("a", top_k=2)

