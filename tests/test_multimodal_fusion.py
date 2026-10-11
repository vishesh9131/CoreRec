"""corerec.multimodal fusion strategies (#79: ~23% covered)."""

import pytest
import torch

from corerec.multimodal.fusion_strategies import (BilinearFusion, ConcatFusion, GatedFusion,
                                                  MultiModalFusion, WeightedFusion)

DIMS = {"text": 12, "image": 20}


def _inputs(batch=3):
    g = torch.Generator().manual_seed(0)
    return {"text": torch.randn(batch, 12, generator=g), "image": torch.randn(batch, 20, generator=g)}


@pytest.mark.parametrize("strategy", ["concat", "weighted", "attention", "gated"])
def test_every_strategy_fuses_to_the_output_dim(strategy):
    torch.manual_seed(0)
    fusion = MultiModalFusion(DIMS, output_dim=8, strategy=strategy).eval()
    assert fusion(_inputs()).shape == (3, 8)


@pytest.mark.parametrize("strategy", ["concat", "weighted", "attention", "gated"])
def test_output_does_not_depend_on_the_order_of_the_input_dict(strategy):
    """concat used the caller's dict order, so the same inputs gave different outputs."""
    torch.manual_seed(0)
    fusion = MultiModalFusion(DIMS, output_dim=8, strategy=strategy).eval()
    x = _inputs()
    reordered = {"image": x["image"], "text": x["text"]}
    assert torch.allclose(fusion(x), fusion(reordered), atol=1e-6)


@pytest.mark.parametrize("strategy", ["concat", "gated"])
def test_strategies_that_need_every_modality_say_which_is_missing(strategy):
    fusion = MultiModalFusion(DIMS, output_dim=8, strategy=strategy)
    with pytest.raises(ValueError, match=r"missing modalities \['image'\]"):
        fusion({"text": torch.randn(2, 12)})


def test_weighted_fusion_skips_a_missing_first_modality():
    """It indexed embeddings[modalities[0]] for the output shape and raised KeyError."""
    w = WeightedFusion(["text", "image"], embedding_dim=4)
    out = w({"image": torch.ones(2, 4)})
    assert torch.allclose(out, torch.full((2, 4), 0.5))  # softmax of equal weights


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        MultiModalFusion(DIMS, output_dim=8, strategy="tucker")


def test_standalone_modules():
    x = _inputs()
    assert ConcatFusion(DIMS, output_dim=5)(x).shape == (3, 5)
    same = {"a": torch.randn(3, 6), "b": torch.randn(3, 6)}
    assert GatedFusion(["a", "b"], embedding_dim=6)(same).shape == (3, 6)
    assert BilinearFusion(12, 20, 7)(x["text"], x["image"]).shape == (3, 7)
