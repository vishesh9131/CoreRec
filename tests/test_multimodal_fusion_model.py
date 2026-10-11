"""corerec.multimodal.fusion_model (#79: 18% covered).

The real encoders download DistilBERT and ViT, so they're swapped for a stub
that returns fixed-size vectors; everything after encoding is the real code.
"""

import pytest
import torch
import torch.nn as nn

import corerec.multimodal.fusion_model as fm


class _StubEncoder(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        self.dim = config["embedding_dim"]

    def encode(self, x):
        return torch.ones(len(x), self.dim)


@pytest.fixture
def make_model(monkeypatch):
    monkeypatch.setattr(fm, "TextEncoder", _StubEncoder)
    monkeypatch.setattr(fm, "VisionEncoder", _StubEncoder)

    def make(fusion_type="concat"):
        torch.manual_seed(0)
        return fm.MultimodalFusionModel("m", {
            "text_encoder_config": {"embedding_dim": 8},
            "vision_encoder_config": {"embedding_dim": 6},
            "fusion_type": fusion_type,
            "fusion_output_dim": 4,
            "prediction_hidden_dims": [4],
        })
    return make


def _batch(n=6):
    g = torch.Generator().manual_seed(0)
    return {"text_features": torch.randn(n, 8, generator=g),
            "image_features": torch.randn(n, 6, generator=g),
            "labels": torch.tensor([1, 0] * (n // 2))}


@pytest.mark.parametrize("fusion_type", ["concat", "attention", "gating"])
def test_every_fusion_type_scores_one_value_per_row(make_model, fusion_type):
    model = make_model(fusion_type).eval()
    assert model(_batch()).shape == (6,)


def test_unknown_fusion_type_is_rejected(make_model):
    with pytest.raises(ValueError, match="Unsupported fusion type"):
        make_model("sum")


def test_raw_inputs_go_through_the_encoders(make_model):
    model = make_model().eval()
    out = model({"text": ["a", "b"], "images": [object(), object()]})
    assert out.shape == (2,)


def test_a_missing_modality_is_filled_with_zeros(make_model):
    model = make_model().eval()
    b = _batch()
    zeros = {"text_features": b["text_features"], "image_features": torch.zeros(6, 6)}
    assert torch.equal(model({"text_features": b["text_features"]}), model(zeros))


def test_a_batch_with_no_modality_is_rejected(make_model):
    with pytest.raises(ValueError, match="at least one modality"):
        make_model()({"labels": torch.ones(2)})


def test_train_step_leaves_the_callers_batch_alone(make_model):
    """train_step popped 'labels', so a second epoch over the same batches raised KeyError."""
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    batches = [_batch(), _batch()]
    for _ in range(2):
        for b in batches:
            model.train_step(b, opt)
    assert all("labels" in b for b in batches)
    model.eval()
    assert "labels" in batches[0] and "val_loss" in model.validate_step(batches[0])
    assert "labels" in batches[0]


def test_integer_labels_train(make_model):
    """0/1 int labels made BCEWithLogitsLoss raise 'Float can't be cast to Long'."""
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    b = _batch()
    assert b["labels"].dtype == torch.int64
    first = model.train_step(b, opt)["loss"]
    for _ in range(30):
        last = model.train_step(b, opt)["loss"]
    assert last < first


def test_validate_step_reports_loss_and_accuracy(make_model):
    model = make_model().eval()
    out = model.validate_step(_batch())
    assert set(out) == {"val_loss", "val_accuracy"}
    assert 0.0 <= out["val_accuracy"] <= 1.0


def test_trains_through_corerec_trainer(make_model, tmp_path):
    from corerec.trainer.trainer import Trainer

    model = make_model("gating")
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=0.01), device=torch.device("cpu"),
                      checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "logs"))
    batches = [_batch(), _batch()]
    trainer.train(batches, val_loader=batches, epochs=2)
    assert all("labels" in b for b in batches)
