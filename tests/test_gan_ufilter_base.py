import pytest
import torch
import numpy as np
from pathlib import Path

from corerec.engines.unionizedFilterEngine.nn_base.gan_ufilter_base import GAN_ufilter_base


class TestGANUFilter:
    @pytest.fixture
    def sample_data(self):
        # Generate sample data for GAN_UFilter model
        users_data = {f'user_{i}': np.random.rand(20) for i in range(10)}
        items_data = {f'item_{i}': np.random.rand(20) for i in range(20)}
        interactions = {}
        for i in range(10):
            user_id = f'user_{i}'
            interactions[user_id] = []
            for j in range(5):
                item_id = f'item_{np.random.randint(0, 20)}'
                interactions[user_id].append(item_id)
        
        return {
            'users': users_data,
            'items': items_data,
            'interactions': interactions
        }
    
    @pytest.fixture
    def model(self):
        return GAN_ufilter_base(
            noise_dim=32,
            gen_hidden_dims=[64, 32],
            disc_hidden_dims=[64, 32],
            num_epochs=2,
            verbose=False
        )
    
    def test_initialization(self, model):
        assert model.name == "GAN_UFilter"
        assert model.noise_dim == 32
        assert model.gen_hidden_dims == [64, 32]
        assert model.disc_hidden_dims == [64, 32]
        assert not model.is_fitted
    
    def test_fit(self, model, sample_data):
        fitted_model = model.fit(sample_data)
        assert fitted_model.is_fitted
        assert hasattr(fitted_model, 'user_map')
        assert hasattr(fitted_model, 'item_map')
        assert hasattr(fitted_model, 'generator')
    
    def test_predict(self, model, sample_data):
        model.fit(sample_data)
        user_id = list(sample_data['users'].keys())[0]
        predictions = model.predict(user_id)
        assert len(predictions) == len(sample_data['items'])
        assert all(0 <= p <= 1 for p in predictions)
    
    def test_recommend(self, model, sample_data):
        model.fit(sample_data)
        user_id = list(sample_data['users'].keys())[0]
        recommendations = model.recommend(user_id, top_n=3)
        assert len(recommendations) == 3
        assert all(0 <= score <= 1 for _, score in recommendations)
    
    def test_save_load(self, model, sample_data, tmp_path):
        model.fit(sample_data)
        save_path = tmp_path / "gan_ufilter_model.pt"
        model.save(str(save_path))
        
        loaded_model = GAN_ufilter_base.load(str(save_path))
        assert loaded_model.is_fitted
        assert loaded_model.name == model.name
        assert loaded_model.noise_dim == model.noise_dim 