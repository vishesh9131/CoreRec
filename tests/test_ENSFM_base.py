import pytest
import torch
import numpy as np
from pathlib import Path

from corerec.engines.unionizedFilterEngine.nn_base.ENSFM_base import ENSFM_base


class TestENSFM:
    @pytest.fixture
    def sample_data(self):
        # Generate sample data for ENSFM model
        data = []
        for i in range(100):
            data.append(
                {
                    "user_id": f"user_{i % 10}",
                    "item_id": f"item_{i % 20}",
                    "label": float(np.random.randint(0, 2)),
                }
            )
        return data

    @pytest.fixture
    def model(self):
        return ENSFM_base(
            embed_dim=16,
            hidden_dims=[
                64,
                32],
            dropout=0.1,
            num_epochs=2,
            verbose=False)

    def test_initialization(self, model):
        assert model.name == "ENSFM"
        assert model.embed_dim == 16
        assert model.hidden_dims == [64, 32]
        assert model.dropout == 0.1
        assert not model.is_fitted

    def test_fit(self, model, sample_data):
        fitted_model = model.fit(sample_data)
        assert fitted_model.is_fitted
        assert hasattr(fitted_model, "user_map")
        assert hasattr(fitted_model, "item_map")
        assert hasattr(fitted_model, "model")

    def test_predict(self, model, sample_data):
        model.fit(sample_data)
        prediction = model.predict(
            sample_data[0]["user_id"],
            sample_data[0]["item_id"])
        assert 0 <= prediction <= 1

    def test_recommend(self, model, sample_data):
        model.fit(sample_data)
        user_id = sample_data[0]["user_id"]
        item_pool = [{"item_id": f"item_{i}"} for i in range(5)]
        recommendations = model.recommend(
            user_id, item_pool=item_pool, top_n=3)
        assert len(recommendations) == 3
        assert all(0 <= score <= 1 for _, score in recommendations)

    def test_save_load(self, model, sample_data, tmp_path):
        model.fit(sample_data)
        save_path = tmp_path / "ensfm_model.pt"
        model.save(str(save_path))

        loaded_model = ENSFM_base.load(str(save_path))
        assert loaded_model.is_fitted
        assert loaded_model.name == model.name
        assert loaded_model.embed_dim == model.embed_dim
