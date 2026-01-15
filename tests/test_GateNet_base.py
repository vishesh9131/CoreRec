import pytest
import torch
import numpy as np
from pathlib import Path

from corerec.engines.collaborative.nn_base.GateNet_base import GateNet_base


class TestGateNet:
    @pytest.fixture
    def sample_data(self):
        # Generate sample data for GateNet model
        data = []
        for i in range(100):
            data.append(
                {
                    "user_id": f"user_{i % 10}",
                    "item_id": f"item_{i % 20}",
                    "category": f"cat_{i % 5}",
                    "label": float(np.random.randint(0, 2)),
                }
            )
        return data

    @pytest.fixture
    def model(self):
        return GateNet_base(
            embed_dim=16,
            mlp_dims=[
                64,
                32],
            num_epochs=2,
            verbose=False)

    def test_initialization(self, model):
        assert model.name == "GateNet"
        assert model.embed_dim == 16
        assert model.mlp_dims == [64, 32]
        assert not model.is_fitted

    def test_fit(self, model, sample_data):
        fitted_model = model.fit(sample_data)
        assert fitted_model.is_fitted
        assert hasattr(fitted_model, "field_mapping")
        assert hasattr(fitted_model, "model")

    def test_predict(self, model, sample_data):
        model.fit(sample_data)
        features = {
            "user_id": sample_data[0]["user_id"],
            "item_id": sample_data[0]["item_id"],
            "category": sample_data[0]["category"],
        }
        prediction = model.predict(features)
        assert 0 <= prediction <= 1

    def test_recommend(self, model, sample_data):
        model.fit(sample_data)
        user_features = {"user_id": sample_data[0]["user_id"]}
        item_pool = [{"item_id": f"item_{i}", "category": f"cat_{i % 5}"}
                     for i in range(5)]
        recommendations = model.recommend(
            user_features, item_pool=item_pool, top_n=3)
        assert len(recommendations) == 3
        assert all(0 <= score <= 1 for _, score in recommendations)

    def test_save_load(self, model, sample_data, tmp_path):
        model.fit(sample_data)
        save_path = tmp_path / "gatenet_model.pt"
        model.save(str(save_path))

        loaded_model = GateNet_base.load(str(save_path))
        assert loaded_model.is_fitted
        assert loaded_model.name == model.name
        assert loaded_model.embed_dim == model.embed_dim
