import pytest
from corerec.engines import SASRec


@pytest.mark.parametrize("user,items", [("u", [30, 10, 20]), ("01", ["01", "1", "002"])])
def test_sasrec_index_and_id_types_survive_persistence(tmp_path, user, items):
    model = SASRec(hidden_units=8, num_heads=1, num_blocks=1, max_seq_length=4,
                   epochs=1, device="cpu", verbose=False, user_cooling=True).fit(
                       [user, user, "other", "other"], [items[0], items[1], items[1], items[2]])
    assert model.item_to_index == model.items_index.as_dict()
    assert model.items_index.offset == 1
    model.save(tmp_path / "m")
    restored = SASRec.load(tmp_path / "m", device="cpu")
    assert restored.item_to_index == model.item_to_index
    assert restored.items_index.as_dict() == model.items_index.as_dict()
    assert restored.user_sequences == model.user_sequences
    assert restored.user_cooling_weights == model.user_cooling_weights
    assert restored.recommend(user, top_k=3) == model.recommend(user, top_k=3)
