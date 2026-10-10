"""JSON pair lists retain ID types, including distinct numeric-looking strings."""
import pytest

from corerec import engines
from corerec.api.bundle_helpers import dict_from_pairs, load_feature_map, load_map_state


@pytest.mark.parametrize("mapping", [{"01": 0, "1": 1}, {0: "01", 1: "1"}])
def test_pair_helpers_preserve_typed_keys_and_values(mapping):
    entries = [list(pair) for pair in mapping.items()]
    assert dict_from_pairs(entries) == mapping
    assert load_map_state({"ids_pairs": entries}, "ids")["ids"] == mapping
    assert load_feature_map({"feature_map_entries": [["item", entries]]}) == {"item": mapping}
    assert dict_from_pairs([["1", "2"]], coerce_numeric=True) == {1: 2}
    assert dict_from_pairs([["1", "02"]], int_keys=True) == {1: "02"}


@pytest.mark.parametrize("name,params", [
    ("DCN", {"embedding_dim": 4, "deep_layers": [4]}),
    ("DeepFM", {"embedding_dim": 4, "hidden_layers": [4]}),
    ("TwoTower", {"embedding_dim": 4, "hidden_dims": [4]}),
    ("LightGCN", {"n_factors": 4, "n_layers": 1}),
])
def test_numeric_string_ids_survive_neural_bundle_loading(name, params, tmp_path):
    cls = getattr(engines, name)
    model = cls(epochs=1, batch_size=4, device="cpu", verbose=False, **params)
    users = ["01", "01", "1", "1", "02", "02"]
    items = ["010", "020", "020", "030", "010", "030"]
    model.fit(users, items, [1.] * len(users))
    path = tmp_path / name
    model.save(path)
    restored = cls.load(path)
    for attribute in ("user_map", "item_map", "user_id_map", "item_id_map", "feature_map"):
        if hasattr(model, attribute):
            assert getattr(restored, attribute) == getattr(model, attribute)
    assert restored.predict("01", "010") == pytest.approx(model.predict("01", "010"))
    assert restored.recommend("01", top_k=1) == model.recommend("01", top_k=1)


def test_tfidf_bundle_preserves_string_ids_and_numeric_text(tmp_path):
    model = engines.TFIDFRecommender().fit(["01", "1", "02"], {"01": "001", "1": "002", "02": "003"})
    path = tmp_path / "tfidf"
    model.save(path)
    restored = type(model).load(path)
    assert restored.item_to_index == model.item_to_index
    assert restored.index_to_item == model.index_to_item
    assert restored.docs == model.docs
    assert restored.predict(None, "01") == model.predict(None, "01")
