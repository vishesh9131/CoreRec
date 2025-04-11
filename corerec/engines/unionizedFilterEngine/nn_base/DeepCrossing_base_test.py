import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
from corerec.engines.unionizedFilterEngine.nn_base.DeepCrossing_base import (
    DeepCrossing_base, ResidualUnit
)


class TestResidualUnit(unittest.TestCase):
    """
    Test suite for ResidualUnit module.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a residual unit
        self.input_dim = 64
        self.batch_size = 32
        self.residual_unit = ResidualUnit(self.input_dim)
    
    def test_forward_pass(self):
        """Test forward pass through the residual unit."""
        # Create a random input tensor
        x = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        output = self.residual_unit(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        
        # Check that output is different from input (residual connection)
        self.assertFalse(torch.allclose(x, output))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the residual unit."""
        # Create a random input tensor that requires grad
        x = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        
        # Forward pass
        output = self.residual_unit(x)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        
        # Check that gradients are computed for all parameters
        for name, param in self.residual_unit.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertGreater(param.grad.abs().sum().item(), 0)


class TestDeepCrossing(unittest.TestCase):
    """
    Test suite for DeepCrossing model.
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory for model saving/loading
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate synthetic data
        self.users = [f'u{i}' for i in range(100)]
        self.items = [f'i{i}' for i in range(50)]
        
        # Generate interactions with features
        self.interactions = []
        for user in self.users[:80]:  # Use 80% users for training
            # Each user interacts with 5-10 items
            num_items = np.random.randint(5, 11)
            item_indices = np.random.choice(range(len(self.items)), size=num_items, replace=False)
            for item_idx in item_indices:
                item = self.items[item_idx]
                # Add features
                features = {
                    'category': np.random.choice(['electronics', 'books', 'clothing']),
                    'price': np.random.uniform(10, 200),
                    'rating': np.random.randint(1, 6),
                    'is_new': np.random.choice([True, False]),
                    'discount': np.random.uniform(0, 0.5)
                }
                self.interactions.append((user, item, features))
        
        # Create test model
        self.model = DeepCrossing_base(
            name="TestDeepCrossing",
            embedding_dim=32,
            hidden_units=[64, 32],
            num_residual_units=2,
            dropout=0.2,
            batch_size=64,
            learning_rate=0.001,
            num_epochs=2,
            seed=42
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test model initialization."""
        # Check that model has correct attributes
        self.assertEqual(self.model.name, "TestDeepCrossing")
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_units, [64, 32])
        self.assertEqual(self.model.num_residual_units, 2)
        self.assertEqual(self.model.dropout, 0.2)
        self.assertEqual(self.model.batch_size, 64)
        self.assertEqual(self.model.learning_rate, 0.001)
        self.assertEqual(self.model.num_epochs, 2)
        self.assertEqual(self.model.seed, 42)
        
        # Check that model is not fitted yet
        self.assertFalse(self.model.is_fitted)
    
    def test_fit(self):
        """Test model fitting."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Check that model is fitted
        self.assertTrue(self.model.is_fitted)
        
        # Check that user and item maps are created
        self.assertGreater(len(self.model.user_map), 0)
        self.assertGreater(len(self.model.item_map), 0)
        
        # Check that feature names and mappings are created
        self.assertGreater(len(self.model.feature_names), 0)
        self.assertGreater(len(self.model.feature_mappings), 0)
        
        # Check that all users and items in interactions are in maps
        for user, item, _ in self.interactions:
            self.assertIn(user, self.model.user_map)
            self.assertIn(item, self.model.item_map)
        
        # Check that model has loss history
        self.assertGreater(len(self.model.train_loss_history), 0)
    
    def test_predict(self):
        """Test prediction."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get a user-item-features tuple from interactions
        user, item, features = self.interactions[0]
        
        # Make prediction
        prediction = self.model.predict(user, item, features)
        
        # Check that prediction is a float
        self.assertIsInstance(prediction, float)
        
        # Check that prediction is within expected range (after sigmoid)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        
        # Test prediction for non-existent user/item
        pred_new_user = self.model.predict("non_existent_user", item, features)
        pred_new_item = self.model.predict(user, "non_existent_item", features)
        
        # Should return 0.0 for unknown users/items
        self.assertEqual(pred_new_user, 0.0)
        self.assertEqual(pred_new_item, 0.0)
        
        # Test prediction with missing features (should use defaults)
        pred_missing_features = self.model.predict(user, item, {})
        self.assertIsInstance(pred_missing_features, float)
    
    def test_recommend(self):
        """Test recommendation generation."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get recommendation for a user
        user = self.users[0]
        recommendations = self.model.recommend(user, top_n=5)
        
        # Check that recommendations is a list of (item, score) tuples
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)
        for item, score in recommendations:
            self.assertIn(item, self.model.item_map)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Check that scores are in descending order
        scores = [score for _, score in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Test with exclude_seen=False
        recommendations_with_seen = self.model.recommend(user, top_n=5, exclude_seen=False)
        self.assertEqual(len(recommendations_with_seen), 5)
        
        # Test with additional features
        features = {
            'category': 'electronics',
            'price': 100,
            'is_new': True
        }
        recommendations_with_features = self.model.recommend(user, top_n=5, features=features)
        self.assertEqual(len(recommendations_with_features), 5)
        
        # Test for non-existent user
        recommendations_new_user = self.model.recommend("non_existent_user", top_n=5)
        self.assertEqual(recommendations_new_user, [])
    
    def test_save_load(self):
        """Test model saving and loading."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Save model
        save_path = os.path.join(self.temp_dir, "model.pt")
        self.model.save(save_path)
        
        # Check that file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load model
        loaded_model = DeepCrossing_base.load(save_path)
        
        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.name, self.model.name)
        self.assertEqual(loaded_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(loaded_model.hidden_units, self.model.hidden_units)
        self.assertEqual(loaded_model.num_residual_units, self.model.num_residual_units)
        self.assertEqual(loaded_model.dropout, self.model.dropout)
        self.assertEqual(loaded_model.seed, self.model.seed)
        
        # Check that user and item maps are the same
        self.assertEqual(loaded_model.user_map, self.model.user_map)
        self.assertEqual(loaded_model.item_map, self.model.item_map)
        self.assertEqual(loaded_model.feature_names, self.model.feature_names)
        
        # Check that predictions are the same
        user, item, features = self.interactions[0]
        original_pred = self.model.predict(user, item, features)
        loaded_pred = loaded_model.predict(user, item, features)
        self.assertAlmostEqual(original_pred, loaded_pred, places=6)
    
    def test_update_incremental(self):
        """Test incremental model update."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Create new interactions with existing and new users/items
        new_users = [f'u{i}' for i in range(100, 110)]
        new_items = [f'i{i}' for i in range(50, 55)]
        
        new_interactions = []
        # Existing user, existing item
        new_interactions.append((self.users[0], self.items[0], {
            'category': 'electronics',
            'price': 150,
            'rating': 5,
            'is_new': True,
            'discount': 0.1
        }))
        # Existing user, new item
        new_interactions.append((self.users[0], new_items[0], {
            'category': 'books',
            'price': 25,
            'rating': 4,
            'is_new': False,
            'discount': 0.2
        }))
        # New user, existing item
        new_interactions.append((new_users[0], self.items[0], {
            'category': 'clothing',
            'price': 50,
            'rating': 3,
            'is_new': True,
            'discount': 0.3
        }))
        # New user, new item
        new_interactions.append((new_users[0], new_items[0], {
            'category': 'electronics',
            'price': 200,
            'rating': 2,
            'is_new': False,
            'discount': 0.4
        }))
        
        # Update model
        self.model.update_incremental(new_interactions, new_users, new_items)
        
        # Check that new users and items are added to mappings
        for user in new_users:
            self.assertIn(user, self.model.user_map)
        
        for item in new_items:
            self.assertIn(item, self.model.item_map)
        
        # Check that we can predict for new user-item pairs
        prediction = self.model.predict(new_users[0], new_items[0], {
            'category': 'electronics',
            'price': 200,
            'rating': 2,
            'is_new': False,
            'discount': 0.4
        })
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get feature importance
        importance = self.model.export_feature_importance()
        
        # Check that importance contains entries for all features
        self.assertIn('user', importance)
        self.assertIn('item', importance)
        for feature in self.model.feature_names:
            self.assertIn(feature, importance)
        
        # Check that importance values are non-negative and sum to 1
        self.assertGreaterEqual(min(importance.values()), 0.0)
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)
    
    def test_hooks(self):
        """Test hook registration and activation retrieval."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Register hook for a layer
        success = self.model.register_hook('model.user_embedding')
        self.assertTrue(success)
        
        # Make a prediction to trigger the hook
        user, item, features = self.interactions[0]
        self.model.predict(user, item, features)
        
        # Get activation
        activation = self.model.get_activation('model.user_embedding')
        self.assertIsNotNone(activation)
        self.assertIsInstance(activation, torch.Tensor)


if __name__ == '__main__':
    unittest.main() 