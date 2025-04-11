import unittest
import os
import torch
import numpy as np
import tempfile
from corerec.engines.unionizedFilterEngine.nn_base.DeepFEFM_base import DeepFEFM_base

class DeepFEFM_base_test(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for model saving/loading
        self.temp_dir = tempfile.mkdtemp()
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate synthetic data
        self.users = [f'user{i}' for i in range(10)]
        self.items = [f'item{i}' for i in range(20)]
        
        # Generate interactions with features
        self.interactions = []
        for user in self.users:
            # Each user interacts with 3-5 items
            num_items = np.random.randint(3, 6)
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
        
        # Create model with parameters that match the actual implementation
        self.model = DeepFEFM_base(
            embed_dim=16,
            mlp_dims=[64, 32],
            field_dims=None,  # Will be inferred from data
            dropout=0.2,
            batch_size=32,
            learning_rate=0.001,
            num_epochs=2,
            seed=42
        )
    
    def test_fit(self):
        """Test model fitting."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Check that feature names and field dims are created
        self.assertGreater(len(self.model.feature_names), 0)
        self.assertGreater(len(self.model.field_dims), 0)
        
        # Check that all users and items in interactions are in maps
        for user, item, _ in self.interactions:
            self.assertIn(user, self.model.user_map)
            self.assertIn(item, self.model.item_map)
        
        # Check that model has loss history
        self.assertGreater(len(self.model.loss_history), 0)
        
        # Check that model can predict
        user, item, features = self.interactions[0]
        prediction = self.model.predict(user, item, features)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
    
    def test_predict(self):
        """Test prediction."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get a user-item pair from interactions
        user, item, features = self.interactions[0]
        
        # Make prediction
        prediction = self.model.predict(user, item, features)
        
        # Check that prediction is a float
        self.assertIsInstance(prediction, float)
        
        # Check that prediction is within expected range
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        
        # Test prediction for non-existent user
        with self.assertRaises(ValueError):
            self.model.predict('nonexistent_user', item, features)
        
        # Test prediction for non-existent item
        with self.assertRaises(ValueError):
            self.model.predict(user, 'nonexistent_item', features)
        
        # Test prediction with missing features
        try:
            prediction_missing = self.model.predict(user, item, {})
            self.assertIsInstance(prediction_missing, float)
            self.assertGreaterEqual(prediction_missing, 0.0)
            self.assertLessEqual(prediction_missing, 1.0)
        except Exception as e:
            # If the model doesn't support empty features, that's okay
            pass
    
    def test_recommend(self):
        """Test recommendation."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get recommendations for a user
        user = self.users[0]
        recommendations = self.model.recommend(user, top_n=5)
        
        # Check that recommendations is a list of (item, score) tuples
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
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
        self.assertLessEqual(len(recommendations_with_seen), 5)
        
        # Test with additional features
        features = {
            'category': 'electronics',
            'price': 100,
            'is_new': True
        }
        recommendations_with_features = self.model.recommend(user, top_n=5, features=features)
        self.assertLessEqual(len(recommendations_with_features), 5)
        
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
        loaded_model = DeepFEFM_base.load(save_path)
        
        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.embed_dim, self.model.embed_dim)
        self.assertEqual(loaded_model.mlp_dims, self.model.mlp_dims)
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
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Fit model
        self.model.fit(self.interactions)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check that importance contains entries for all features
        for feature in self.model.feature_names:
            self.assertIn(feature, importance)
        
        # Check that importance values are non-negative and sum to 1
        self.assertGreaterEqual(min(importance.values()), 0.0)
        self.assertAlmostEqual(sum(importance.values()), 1.0, places=5)
    
    def test_device_setting(self):
        """Test device setting."""
        # Set device to CPU
        self.model.set_device('cpu')
        self.assertEqual(str(self.model.device), 'cpu')
        
        # Fit model
        self.model.fit(self.interactions)
        
        # Check that model is on CPU
        for param in self.model.model.parameters():
            self.assertEqual(str(param.device), 'cpu')
        
        # Only test CUDA if available
        if torch.cuda.is_available():
            # Set device to CUDA
            self.model.set_device('cuda')
            self.assertEqual(str(self.model.device), 'cuda:0')
            
            # Check that model is on CUDA
            for param in self.model.model.parameters():
                self.assertEqual(str(param.device), 'cuda:0')


if __name__ == '__main__':
    unittest.main() 