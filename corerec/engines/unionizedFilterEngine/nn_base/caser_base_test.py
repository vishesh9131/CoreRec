import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from corerec.engines.unionizedFilterEngine.nn_base.caser_base import (
    Caser_base, CaserModel, HorizontalConvolution, VerticalConvolution, HookManager
)


class TestCaserComponents(unittest.TestCase):
    """Test individual components of the Caser model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.seq_len = 10
        self.embedding_dim = 16
        self.vocab_size = 100
        self.num_h_filters = 8
        self.num_v_filters = 4
    
    def test_horizontal_convolution(self):
        """Test the HorizontalConvolution module."""
        h_conv = HorizontalConvolution(self.num_h_filters, self.embedding_dim)
        x = torch.rand((self.batch_size, self.seq_len, self.embedding_dim))
        output = h_conv(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_h_filters * 2)  # 2 window sizes
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for conv in h_conv.conv_layers:
            self.assertIsNotNone(conv.weight.grad)
    
    def test_vertical_convolution(self):
        """Test the VerticalConvolution module."""
        v_conv = VerticalConvolution(self.num_v_filters, self.seq_len)
        x = torch.rand((self.batch_size, self.seq_len, self.embedding_dim))
        output = v_conv(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_v_filters * self.embedding_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(v_conv.conv.weight.grad)
    
    def test_caser_model(self):
        """Test the CaserModel."""
        model = CaserModel(
            num_items=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.seq_len,
            num_h_filters=self.num_h_filters,
            num_v_filters=self.num_v_filters
        )
        
        # Create input tensor
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.vocab_size))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_hook_manager(self):
        """Test the HookManager."""
        hook_manager = HookManager()
        model = CaserModel(
            num_items=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.seq_len,
            num_h_filters=self.num_h_filters,
            num_v_filters=self.num_v_filters
        )
        
        # Register hook
        hook_registered = hook_manager.register_hook(model, 'item_embedding')
        self.assertTrue(hook_registered)
        
        # Forward pass to trigger hook
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        model(x)
        
        # Check that activation was captured
        activation = hook_manager.get_activation('item_embedding')
        self.assertIsNotNone(activation)
        self.assertEqual(activation.shape, (self.batch_size, self.seq_len, self.embedding_dim))
        
        # Remove hook
        hook_removed = hook_manager.remove_hook('item_embedding')
        self.assertTrue(hook_removed)
        
        # Clear activations
        hook_manager.clear_activations()
        self.assertEqual(hook_manager.activations, {})


class TestCaserBase(unittest.TestCase):
    """Test the Caser_base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Model parameters
        self.config = {
            'embedding_dim': 16,
            'num_h_filters': 8,
            'num_v_filters': 4,
            'max_seq_len': 5,
            'dropout_rate': 0.2,
            'batch_size': 4,
            'num_epochs': 2,
            'device': 'cpu',
            'negative_samples': 1
        }
        
        # Create test data
        self.num_users = 10
        self.num_items = 20
        self.user_ids = [f'user_{i}' for i in range(self.num_users)]
        self.item_ids = [f'item_{i}' for i in range(self.num_items)]
        
        # Create interactions
        self.interactions = []
        for i in range(self.num_users):
            items_per_user = np.random.randint(5, 15)
            items = np.random.choice(self.num_items, size=items_per_user, replace=False)
            for j, item_idx in enumerate(items):
                self.interactions.append((f'user_{i}', f'item_{item_idx}', i * 100 + j))
        
        # Create model instance
        self.model = Caser_base(name="TestCaser", config=self.config, verbose=False)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "TestCaser")
        self.assertEqual(self.model.config['embedding_dim'], 16)
        self.assertEqual(self.model.config['num_h_filters'], 8)
        self.assertEqual(self.model.config['num_v_filters'], 4)
        self.assertEqual(self.model.device, torch.device('cpu'))
        self.assertFalse(self.model.is_fitted)
    
    def test_prepare_sequences(self):
        """Test prepare_sequences method."""
        sequences = self.model._prepare_sequences(self.interactions)
        
        # Check that sequences have been created
        self.assertIsInstance(sequences, list)
        self.assertEqual(len(sequences), self.num_users)
        
        # Check that uid_map and iid_map have been created
        self.assertEqual(len(self.model.uid_map), self.num_users)
        self.assertEqual(len(self.model.iid_map), self.num_items)
        
        # Check that all users and items are in the mappings
        for user_id in self.user_ids:
            self.assertIn(user_id, self.model.uid_map)
        
        for item_id in self.item_ids:
            self.assertIn(item_id, self.model.iid_map)
        
        # Check that sequences contain valid item indices
        for seq in sequences:
            for item_idx in seq:
                self.assertLess(item_idx, self.num_items + 1)  # +1 for padding
                self.assertGreaterEqual(item_idx, 1)  # item indices start at 1
    
    def test_fit(self):
        """Test fit method."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Check that model has been trained
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.model)
        
        # Check that loss history has been recorded
        self.assertIsInstance(self.model.loss_history, list)
        self.assertEqual(len(self.model.loss_history), self.config['num_epochs'])
        
        # Check that user_sequences have been created
        self.assertIsInstance(self.model.user_sequences, list)
        self.assertEqual(len(self.model.user_sequences), self.num_users)
    
    def test_predict(self):
        """Test predict method."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Make predictions for all users and items
        for user_id in self.user_ids[:2]:  # Test a subset to keep it fast
            for item_id in self.item_ids[:2]:
                score = self.model.predict(user_id, item_id)
                
                # Check that score is a float
                self.assertIsInstance(score, float)
                
                # Check that score is in a reasonable range
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_recommend(self):
        """Test recommend method."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Get recommendations for a user
        user_id = self.user_ids[0]
        recommendations = self.model.recommend(user_id, top_n=5)
        
        # Check that recommendations are returned as a list of (item_id, score) tuples
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        for item_id, score in recommendations:
            # Check that item_id is in our item list
            self.assertIn(item_id, self.item_ids)
            
            # Check that score is in a reasonable range
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_save_load(self):
        """Test save and load methods."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Fit the model
            self.model.fit(self.interactions)
            
            # Save the model
            model_path = os.path.join(temp_dir, "caser_model")
            self.model.save(model_path)
            
            # Check that model files have been created
            self.assertTrue(os.path.exists(f"{model_path}.pkl"))
            self.assertTrue(os.path.exists(f"{model_path}.meta"))
            
            # Load the model
            loaded_model = Caser_base.load(f"{model_path}.pkl")
            
            # Check that the loaded model has the same configuration
            self.assertEqual(loaded_model.config['embedding_dim'], self.model.config['embedding_dim'])
            self.assertEqual(loaded_model.config['num_h_filters'], self.model.config['num_h_filters'])
            self.assertEqual(loaded_model.config['num_v_filters'], self.model.config['num_v_filters'])
            
            # Check that the loaded model has the same mappings
            self.assertEqual(len(loaded_model.uid_map), len(self.model.uid_map))
            self.assertEqual(len(loaded_model.iid_map), len(self.model.iid_map))
            
            # Make predictions with both models
            user_id = self.user_ids[0]
            item_id = self.item_ids[0]
            
            original_score = self.model.predict(user_id, item_id)
            loaded_score = loaded_model.predict(user_id, item_id)
            
            # Check that predictions are the same
            self.assertAlmostEqual(original_score, loaded_score, places=5)
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_incremental_update(self):
        """Test incremental update."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Create new users and items
        new_users = [f'new_user_{i}' for i in range(3)]
        new_items = [f'new_item_{i}' for i in range(2)]
        
        # Create new interactions
        new_interactions = []
        
        # New users interact with existing items
        for i, user_id in enumerate(new_users):
            for j in range(3):
                item_idx = (i + j) % self.num_items
                item_id = self.item_ids[item_idx]
                new_interactions.append((user_id, item_id, 1000 + i * 10 + j))
        
        # Existing users interact with new items
        for i, user_id in enumerate(self.user_ids[:5]):
            for j, item_id in enumerate(new_items):
                new_interactions.append((user_id, item_id, 2000 + i * 10 + j))
        
        # Update model incrementally
        self.model.update_incremental(new_interactions, new_user_ids=new_users, new_item_ids=new_items)
        
        # Check that model has been updated
        self.assertEqual(len(self.model.user_ids), self.num_users + len(new_users))
        self.assertEqual(len(self.model.item_ids), self.num_items + len(new_items))
        
        # Check that we can make predictions for new users and items
        new_user_id = new_users[0]
        new_item_id = new_items[0]
        
        score = self.model.predict(new_user_id, new_item_id)
        self.assertIsInstance(score, float)
    
    def test_export_embeddings(self):
        """Test export_embeddings method."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Export embeddings
        embeddings = self.model.export_embeddings()
        
        # Check that embeddings are returned as a dictionary
        self.assertIsInstance(embeddings, dict)
        
        # Check that all items have embeddings
        self.assertEqual(len(embeddings), self.num_items)
        
        for item_id in self.item_ids:
            self.assertIn(item_id, embeddings)
            
            # Check that embedding is a list of floats
            embedding = embeddings[item_id]
            self.assertIsInstance(embedding, list)
            self.assertEqual(len(embedding), self.config['embedding_dim'])
    
    def test_set_device(self):
        """Test set_device method."""
        # Set device
        self.model.set_device('cpu')
        
        # Check that device has been set
        self.assertEqual(self.model.device, torch.device('cpu'))
        
        # Set the model
        self.model.fit(self.interactions)
        
        # Check that model is on the correct device
        self.assertEqual(next(self.model.model.parameters()).device, torch.device('cpu'))
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Create another model with the same configuration and seed
        model2 = Caser_base(name="TestCaser2", config=self.config, seed=42, verbose=False)
        model2.fit(self.interactions)
        
        # Check that models have the same weights
        for p1, p2 in zip(self.model.model.parameters(), model2.model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        # Check that predictions are the same
        user_id = self.user_ids[0]
        item_id = self.item_ids[0]
        
        score1 = self.model.predict(user_id, item_id)
        score2 = model2.predict(user_id, item_id)
        
        self.assertAlmostEqual(score1, score2, places=5)


if __name__ == '__main__':
    unittest.main()