import unittest
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import tempfile
from corerec.engines.unionizedFilterEngine.nn_base.DIFM_base import DIFM_base, InterestFusionLayer


class TestInterestFusionLayer(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.attention_dim = 32
        self.batch_size = 16
        self.layer = InterestFusionLayer(self.input_dim, self.attention_dim)

    def test_initialization(self):
        """Test proper initialization of InterestFusionLayer"""
        self.assertIsInstance(self.layer.attention, torch.nn.Sequential)
        self.assertIsInstance(self.layer.fusion, torch.nn.Linear)

    def test_forward_pass(self):
        """Test forward pass of InterestFusionLayer"""
        # Create dummy input tensors
        user_features = torch.randn(self.batch_size, self.input_dim)
        item_features = torch.randn(self.batch_size, self.input_dim)

        # Forward pass
        output = self.layer(user_features, item_features)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))

    def test_attention_weights(self):
        """Test attention weights are between 0 and 1"""
        user_features = torch.randn(self.batch_size, self.input_dim)
        item_features = torch.randn(self.batch_size, self.input_dim)

        # Get attention weights
        combined = torch.cat([user_features, item_features], dim=-1)
        attention_weights = self.layer.attention(combined)

        # Check attention weights range
        self.assertTrue(torch.all(attention_weights >= 0))
        self.assertTrue(torch.all(attention_weights <= 1))


class TestDIFMBase(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.mlp_dims = [128, 64]
        self.field_dims = [100, 200]  # [num_users, num_items]
        self.dropout = 0.1
        self.attention_dim = 32
        self.batch_size = 16
        self.learning_rate = 0.001
        self.num_epochs = 2
        self.seed = 42

        self.model = DIFM_base(
            embed_dim=self.embed_dim,
            mlp_dims=self.mlp_dims,
            field_dims=self.field_dims,
            dropout=self.dropout,
            attention_dim=self.attention_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            seed=self.seed,
        )

        # Create dummy interactions
        self.interactions = [
            (f"user_{i}", f"item_{j}", {"feature": i + j}) for i in range(50) for j in range(50)
        ]

    def test_initialization(self):
        """Test proper initialization of DIFM_base"""
        self.assertEqual(self.model.embed_dim, self.embed_dim)
        self.assertEqual(self.model.mlp_dims, self.mlp_dims)
        self.assertEqual(self.model.field_dims, self.field_dims)
        self.assertEqual(self.model.dropout, self.dropout)
        self.assertEqual(self.model.attention_dim, self.attention_dim)
        self.assertEqual(self.model.batch_size, self.batch_size)
        self.assertEqual(self.model.learning_rate, self.learning_rate)
        self.assertEqual(self.model.num_epochs, self.num_epochs)
        self.assertEqual(self.model.seed, self.seed)

    def test_build_model(self):
        """Test model architecture building"""
        self.model.build_model()

        # Check embeddings
        self.assertIsInstance(self.model.embeddings, torch.nn.ModuleList)
        self.assertEqual(len(self.model.embeddings), len(self.field_dims))

        # Check interest fusion layer
        self.assertIsInstance(self.model.interest_fusion, InterestFusionLayer)

        # Check MLP layers
        self.assertIsInstance(self.model.mlp, torch.nn.ModuleList)

        # Check output layer
        self.assertIsInstance(self.model.output_layer, torch.nn.Linear)
        self.assertIsInstance(self.model.sigmoid, torch.nn.Sigmoid)

    def test_fit(self):
        """Test model training"""
        self.model.fit(self.interactions)

        # Check if model is fitted
        self.assertTrue(self.model.is_fitted)

        # Check if loss history is recorded
        self.assertTrue(len(self.model.loss_history) > 0)

    def test_predict(self):
        """Test prediction functionality"""
        # First fit the model
        self.model.fit(self.interactions)

        # Test prediction for known user and item
        user = "user_1"
        item = "item_1"
        features = {"feature": 2}

        prediction = self.model.predict(user, item, features)

        # Check prediction is between 0 and 1
        self.assertTrue(0 <= prediction <= 1)

        # Test prediction for unknown user
        with self.assertRaises(ValueError):
            self.model.predict("unknown_user", item, features)

        # Test prediction for unknown item
        with self.assertRaises(ValueError):
            self.model.predict(user, "unknown_item", features)

    def test_recommend(self):
        """Test recommendation functionality"""
        # First fit the model
        self.model.fit(self.interactions)

        # Test recommendations for known user
        user = "user_1"
        top_n = 5
        recommendations = self.model.recommend(user, top_n=top_n)

        # Check number of recommendations
        self.assertEqual(len(recommendations), top_n)

        # Check recommendation format
        for item, score in recommendations:
            self.assertIsInstance(item, str)
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

        # Test recommendations for unknown user
        recommendations = self.model.recommend("unknown_user")
        self.assertEqual(len(recommendations), 0)

    def test_prepare_batch(self):
        """Test batch preparation"""
        # First extract features to create mappings
        self.model._extract_features(self.interactions)

        batch = self.interactions[: self.batch_size]
        user_features, item_features, labels = self.model._prepare_batch(batch)

        # Check shapes
        self.assertEqual(user_features.shape, (self.batch_size,))
        self.assertEqual(item_features.shape, (self.batch_size,))
        self.assertEqual(labels.shape, (self.batch_size,))

        # Check labels are all 1
        self.assertTrue(torch.all(labels == 1))

    def test_extract_features(self):
        """Test feature extraction"""
        self.model._extract_features(self.interactions)

        # Check user and item maps
        self.assertTrue(len(self.model.user_map) > 0)
        self.assertTrue(len(self.model.item_map) > 0)

        # Check field dimensions
        self.assertEqual(len(self.model.field_dims), 2)
        self.assertEqual(self.model.field_dims[0], len(self.model.user_map))
        self.assertEqual(self.model.field_dims[1], len(self.model.item_map))

    def test_save_and_load(self):
        """Test model saving and loading"""
        # First fit the model
        self.model.fit(self.interactions)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            # Save model
            self.model.save(filepath)

            # Load model
            loaded_model = DIFM_base.load(filepath)

            # Check if loaded model is fitted
            self.assertTrue(loaded_model.is_fitted)

            # Check if parameters match
            self.assertEqual(loaded_model.embed_dim, self.embed_dim)
            self.assertEqual(loaded_model.mlp_dims, self.mlp_dims)

            # Field dimensions will be different after fitting
            # as they are determined by the actual data
            self.assertTrue(len(loaded_model.field_dims) == 2)
            self.assertTrue(loaded_model.field_dims[0] > 0)
            self.assertTrue(loaded_model.field_dims[1] > 0)

            # Test prediction with loaded model
            user = "user_1"
            item = "item_1"
            features = {"feature": 2}

            original_pred = self.model.predict(user, item, features)
            loaded_pred = loaded_model.predict(user, item, features)

            self.assertAlmostEqual(original_pred, loaded_pred, places=5)

        finally:
            # Clean up
            os.unlink(filepath)


if __name__ == "__main__":
    unittest.main()
