import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from corerec.engines.unionizedFilterEngine.nn_base.autoencoder_cf_base import (
    AutoencoderCFBase, AutoencoderCFModel, Encoder, Decoder, HookManager
)


class TestAutoEncoderComponents(unittest.TestCase):
    """Test individual components of the AutoencoderCF model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.num_items = 100
        self.hidden_dims = [64, 32]
        self.latent_dim = 16
    
    def test_encoder(self):
        """Test the Encoder module."""
        encoder = Encoder(self.num_items, self.hidden_dims + [self.latent_dim], dropout=0.1)
        x = torch.rand(self.batch_size, self.num_items)
        output = encoder(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.latent_dim))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_decoder(self):
        """Test the Decoder module."""
        decoder = Decoder(self.latent_dim, self.hidden_dims[::-1], self.num_items, dropout=0.1)
        z = torch.rand(self.batch_size, self.latent_dim)
        output = decoder(z)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_items))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in decoder.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_autoencoder_model(self):
        """Test the AutoencoderCFModel."""
        model = AutoencoderCFModel(
            num_items=self.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=0.1
        )
        x = torch.rand(self.batch_size, self.num_items)
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_items))
        
        # Check encode and decode methods
        z = model.encode(x)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        
        decoded = model.decode(z)
        self.assertEqual(decoded.shape, (self.batch_size, self.num_items))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_hook_manager(self):
        """Test the HookManager."""
        hooks = HookManager()
        model = AutoencoderCFModel(
            num_items=self.num_items,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=0.1
        )
        
        # Register hook
        success = hooks.register_hook(model, 'encoder')
        self.assertTrue(success)
        
        # Forward pass
        x = torch.rand(self.batch_size, self.num_items)
        output = model(x)
        
        # Check activation
        activation = hooks.get_activation('encoder')
        self.assertIsNotNone(activation)
        self.assertEqual(activation.shape, (self.batch_size, self.latent_dim))
        
        # Remove hook
        success = hooks.remove_hook('encoder')
        self.assertTrue(success)
        
        # Clear activations
        hooks.clear_activations()
        self.assertEqual(len(hooks.activations), 0)


class TestAutoEncoderCFBase(unittest.TestCase):
    """Test the AutoencoderCFBase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test data
        self.num_users = 50
        self.num_items = 100
        
        # Create sparse interaction matrix
        self.interaction_matrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        
        # Add some interactions (about 5% density)
        for _ in range(int(0.05 * self.num_users * self.num_items)):
            u = np.random.randint(0, self.num_users)
            i = np.random.randint(0, self.num_items)
            self.interaction_matrix[u, i] = 1.0
        
        # Convert to CSR for efficient row slicing
        self.interaction_matrix = self.interaction_matrix.tocsr()
        
        # Create user and item IDs
        self.user_ids = [f"user_{i}" for i in range(self.num_users)]
        self.item_ids = [f"item_{i}" for i in range(self.num_items)]
        
        # Initialize the AutoencoderCF model with small dimensions for testing
        self.model = AutoencoderCFBase(
            name="TestAutoencoder",
            trainable=True,
            verbose=True,
            config={
                'hidden_dims': [32, 16],
                'latent_dim': 8,
                'dropout': 0.1,
                'learning_rate': 0.01,
                'weight_decay': 1e-6,
                'batch_size': 16,
                'num_epochs': 2,  # Small number for testing
                'device': 'cpu'
            },
            seed=42
        )
        
        # Create temp directory for saving/loading
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test initialization of AutoencoderCFBase."""
        self.assertEqual(self.model.name, "TestAutoencoder")
        self.assertTrue(self.model.trainable)
        self.assertTrue(self.model.verbose)
        self.assertEqual(self.model.config['hidden_dims'], [32, 16])
        self.assertEqual(self.model.config['latent_dim'], 8)
        self.assertIsNotNone(self.model.hooks)
        self.assertEqual(self.model.version, "1.0.0")
    
    def test_fit_and_recommend(self):
        """Test fit and recommend methods."""
        # Fit the model
        history = self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Check that history contains expected keys
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        
        # Check that model is fitted
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.model.num_users, self.num_users)
        self.assertEqual(self.model.num_items, self.num_items)
        
        # Test recommendation
        user_id = self.user_ids[0]
        recommendations = self.model.recommend(user_id, top_n=5)
        
        # Check recommendations format
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        if len(recommendations) > 0:
            # Check that recommendations are tuples of (item_id, score)
            self.assertIsInstance(recommendations[0], tuple)
            self.assertEqual(len(recommendations[0]), 2)
            
            # Check that scores are in descending order
            scores = [score for _, score in recommendations]
            self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_save_and_load(self):
        """Test save and load methods."""
        # Fit the model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Save the model
        save_path = os.path.join(self.temp_dir, "autoencoder_model")
        self.model.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{save_path}.pkl"))
        self.assertTrue(os.path.exists(f"{save_path}.meta"))
        
        # Load the model
        loaded_model = AutoencoderCFBase.load(f"{save_path}.pkl")
        
        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.name, self.model.name)
        self.assertEqual(loaded_model.num_users, self.model.num_users)
        self.assertEqual(loaded_model.num_items, self.model.num_items)
        self.assertEqual(loaded_model.config['latent_dim'], self.model.config['latent_dim'])
        
        # Test recommendation with loaded model
        user_id = self.user_ids[0]
        recommendations = loaded_model.recommend(user_id, top_n=5)
        self.assertIsInstance(recommendations, list)
    
    def test_register_hook(self):
        """Test register_hook method."""
        # Build model first
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Register hook
        success = self.model.register_hook('encoder')
        self.assertTrue(success)
        
        # Make a prediction to trigger the hook
        user_id = self.user_ids[0]
        self.model.recommend(user_id, top_n=5)
        
        # Check activation
        activation = self.model.hooks.get_activation('encoder')
        self.assertIsNotNone(activation)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        # Set early stopping parameters
        self.model.config['patience'] = 1
        self.model.config['min_delta'] = 0.01
        
        # Fit with early stopping
        history = self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Check that early stopping was triggered
        self.assertLessEqual(len(history['loss']), self.model.config['num_epochs'])
    
    def test_update_incremental(self):
        """Test incremental update functionality."""
        # Fit the initial model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Create new users and items
        new_users = [f"new_user_{i}" for i in range(5)]
        new_items = [f"new_item_{i}" for i in range(3)]
        
        # Create new interaction matrix
        new_num_users = self.num_users + len(new_users)
        new_num_items = self.num_items + len(new_items)
        new_matrix = sp.dok_matrix((new_num_users, new_num_items), dtype=np.float32)
        
        # Copy existing interactions
        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.interaction_matrix[i, j] > 0:
                    new_matrix[i, j] = self.interaction_matrix[i, j]
        
        # Add some new interactions
        for i in range(self.num_users, new_num_users):
            for j in range(self.num_items, new_num_items):
                if np.random.random() < 0.2:  # 20% chance of interaction
                    new_matrix[i, j] = 1.0
        
        # Convert to CSR
        new_matrix = new_matrix.tocsr()
        
        # Update model incrementally
        all_users = self.user_ids + new_users
        all_items = self.item_ids + new_items
        updated_model = self.model.update_incremental(new_matrix, all_users, all_items)
        
        # Check that model was updated
        self.assertEqual(updated_model.num_users, new_num_users)
        self.assertEqual(updated_model.num_items, new_num_items)
        
        # Test recommendation for new user
        new_user_id = new_users[0]
        recommendations = updated_model.recommend(new_user_id, top_n=5)
        self.assertIsInstance(recommendations, list)
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        # Fit model with seed 42
        model1 = AutoencoderCFBase(
            name="TestAutoencoder1",
            config={'seed': 42, 'num_epochs': 2, 'device': 'cpu'},
            seed=42
        )
        model1.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Fit another model with same seed
        model2 = AutoencoderCFBase(
            name="TestAutoencoder2",
            config={'seed': 42, 'num_epochs': 2, 'device': 'cpu'},
            seed=42
        )
        model2.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Get recommendations for both models
        user_id = self.user_ids[0]
        rec1 = model1.recommend(user_id, top_n=5)
        rec2 = model2.recommend(user_id, top_n=5)
        
        # Check that recommendations are the same
        self.assertEqual(len(rec1), len(rec2))
        for i in range(len(rec1)):
            self.assertEqual(rec1[i][0], rec2[i][0])  # Same item
            self.assertAlmostEqual(rec1[i][1], rec2[i][1], places=5)  # Same score
    
    def test_different_loss_functions(self):
        """Test different loss functions."""
        # Test with BCE loss
        self.model.config['loss_function'] = 'bce'
        history_bce = self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Test with MSE loss
        self.model.config['loss_function'] = 'mse'
        history_mse = self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Both should complete without errors
        self.assertIn('loss', history_bce)
        self.assertIn('loss', history_mse)


if __name__ == '__main__':
    unittest.main()