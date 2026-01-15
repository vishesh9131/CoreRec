import pytest
import torch
import numpy as np
from pathlib import Path

from corerec.engines.collaborative.nn_base.GNN_base import GNN_base


class TestGNN:
    @pytest.fixture
    def sample_data(self):
        # Generate sample data for GNN model - user-item interactions
        data = []
        for i in range(100):
            user_id = f"user_{i % 10}"
            item_id = f"item_{i % 20}"
            rating = float(np.random.randint(1, 6))  # Ratings 1-5
            data.append((user_id, item_id, rating))
        return data

    @pytest.fixture
    def model(self):
        return GNN_base(
            embed_dim=32,
            hidden_dims=[
                64,
                32],
            num_epochs=2,
            verbose=False)

    def test_initialization(self, model):
        assert model.name == "GNN"
        assert model.embed_dim == 32
        assert model.hidden_dims == [64, 32]
        assert not model.is_fitted

    def test_fit(self, model, sample_data):
        fitted_model = model.fit(sample_data)
        assert fitted_model.is_fitted
        assert hasattr(fitted_model, "user_map")
        assert hasattr(fitted_model, "item_map")
        assert hasattr(fitted_model, "model")

    def test_predict(self, model, sample_data):
        model.fit(sample_data)
        user_id = sample_data[0][0]
        item_id = sample_data[0][1]
        prediction = model.predict(user_id, item_id)
        assert isinstance(prediction, float)

    def test_recommend(self, model, sample_data):
        model.fit(sample_data)
        user_id = sample_data[0][0]
        recommendations = model.recommend(user_id, top_n=3)
        assert len(recommendations) <= 3  # Could be less if exclude_seen=True
        assert all(isinstance(score, float) for _, score in recommendations)

    def test_save_load(self, model, sample_data, tmp_path):
        model.fit(sample_data)
        save_path = tmp_path / "gnn_model.pt"
        model.save(str(save_path))

        loaded_model = GNN_base.load(str(save_path))
        assert loaded_model.is_fitted
        assert loaded_model.name == model.name
        assert loaded_model.embed_dim == model.embed_dim
