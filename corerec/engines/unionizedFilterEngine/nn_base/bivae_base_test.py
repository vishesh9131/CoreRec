import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from corerec.engines.unionizedFilterEngine.nn_base.bivae_base import (
    BiVAE_base, BIVAE, Encoder, Decoder, HookManager
)


class TestBiVAEComponents(unittest.TestCase):
    """Test individual components of the BiVAE model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.input_dim = 20
        self.hidden_dims = [16, 8]
        self.latent_dim = 4
    
    def test_encoder(self):
        """Test the Encoder module."""
        encoder = Encoder(self.input_dim, self.hidden_dims, self.latent_dim)
        x = torch.rand(self.batch_size, self.input_dim)
        mu, logvar, z = encoder(x)
        
        # Check output shapes
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        
        # Check that output is differentiable
        loss = mu.sum() + logvar.sum() + z.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_decoder(self):
        """Test the Decoder module."""
        decoder = Decoder(self.latent_dim, self.hidden_dims[::-1], self.input_dim)
        z = torch.rand(self.batch_size, self.latent_dim)
        output = decoder(z)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.input_dim))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in decoder.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_bivae_model(self):
        """Test the BIVAE model."""
        model = BIVAE(
            num_users=10,
            num_items=20,
            latent_dim=self.latent_dim,
            encoder_hidden_dims=self.hidden_dims,
            decoder_hidden_dims=self.hidden_dims[::-1]
        )
        
        # Test user encoding
        user_data = torch.rand(self.batch_size, 20)
        user_mu, user_logvar, user_z = model.encode_user(user_data)
        
        # Check user encoding shapes
        self.assertEqual(user_mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(user_logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(user_z.shape, (self.batch_size, self.latent_dim))
        
        # Test item encoding
        item_data = torch.rand(self.batch_size, 10)
        item_mu, item_logvar, item_z = model.encode_item(item_data)
        
        # Check item encoding shapes
        self.assertEqual(item_mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(item_logvar.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(item_z.shape, (self.batch_size, self.latent_dim))
        
        # Test user decoding
        user_recon = model.decode_user(user_z)
        self.assertEqual(user_recon.shape, (self.batch_size, 20))
        
        # Test item decoding
        item_recon = model.decode_item(item_z)
        self.assertEqual(item_recon.shape, (self.batch_size, 10))
        
        # Test forward pass
        user_data = torch.rand(self.batch_size, 20)
        item_data = torch.rand(self.batch_size, 10)
        
        results = model(user_data, item_data)
        user_mu, user_logvar, user_z, user_recon = results[:4]
        item_mu, item_logvar, item_z, item_recon = results[4:]
        
        # Check output shapes
        self.assertEqual(user_recon.shape, (self.batch_size, 20))
        self.assertEqual(item_recon.shape, (self.batch_size, 10))
        
        # Test loss computation
        loss = model.loss_function(user_data, user_recon, user_mu, user_logvar,
                                  item_data, item_recon, item_mu, item_logvar)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Check that loss is differentiable
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)


class TestBiVAEBase(unittest.TestCase):
    """Test the BiVAE_base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.num_users = 10
        self.num_items = 20
        self.latent_dim = 4
        
        # Create user and item IDs
        self.user_ids = [f'user_{i}' for i in range(self.num_users)]
        self.item_ids = [f'item_{i}' for i in range(self.num_items)]
        
        # Create interaction matrix (sparse)
        self.interaction_matrix = sp.lil_matrix((self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                if np.random.random() < 0.2:  # 20% of entries are interactions
                    self.interaction_matrix[i, j] = 1.0
        
        # Convert to CSR format for efficient operations
        self.interaction_matrix = self.interaction_matrix.tocsr()
        
        # Create model
        self.model = BiVAE_base(
            name="TestBiVAE",
            config={
                'latent_dim': self.latent_dim,
                'encoder_hidden_dims': [16, 8],
                'decoder_hidden_dims': [8, 16],
                'batch_size': 4,
                'num_epochs': 2,
                'device': 'cpu',
                'learning_rate': 0.01,
                'beta': 0.1,
                'early_stopping_patience': 2
            },
            trainable=True,
            verbose=False,
            seed=42
        )
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.model.name, "TestBiVAE")
        self.assertEqual(self.model.config['latent_dim'], self.latent_dim)
        self.assertTrue(self.model.trainable)
        self.assertFalse(self.model.verbose)
        self.assertEqual(self.model.seed, 42)
        self.assertFalse(self.model.is_fitted)
    
    def test_build_model(self):
        """Test model building."""
        # Build model
        self.model._build_model(self.num_users, self.num_items)
        
        # Check model components
        self.assertIsInstance(self.model.model, BIVAE)
        self.assertEqual(self.model.model.num_users, self.num_users)
        self.assertEqual(self.model.model.num_items, self.num_items)
        self.assertEqual(self.model.model.latent_dim, self.latent_dim)
    
    def test_fit_recommend(self):
        """Test fitting and recommendation."""
        # Fit model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Check that model is fitted
        self.assertTrue(self.model.is_fitted)
        
        # Test recommendation
        user_id = self.user_ids[0]
        recommendations = self.model.recommend(user_id, top_n=5)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        for item, score in recommendations:
            self.assertIn(item, self.item_ids)
            self.assertIsInstance(score, float)
    
    def test_save_load(self):
        """Test saving and loading the model."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Fit model
            self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
            
            # Save model
            save_path = os.path.join(temp_dir, "bivae_test")
            self.model.save(save_path)
            
            # Load model
            loaded_model = BiVAE_base.load(f"{save_path}.pkl")
            
            # Check that loaded model is the same
            self.assertEqual(loaded_model.name, self.model.name)
            self.assertEqual(loaded_model.config, self.model.config)
            self.assertEqual(loaded_model.num_users, self.model.num_users)
            self.assertEqual(loaded_model.num_items, self.model.num_items)
            
            # Test recommendation with loaded model
            user_id = self.user_ids[0]
            orig_recommendations = self.model.recommend(user_id, top_n=5)
            loaded_recommendations = loaded_model.recommend(user_id, top_n=5)
            
            # Check that recommendations are the same
            self.assertEqual(len(orig_recommendations), len(loaded_recommendations))
            for i in range(len(orig_recommendations)):
                self.assertEqual(orig_recommendations[i][0], loaded_recommendations[i][0])
                self.assertAlmostEqual(orig_recommendations[i][1], loaded_recommendations[i][1], places=5)
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def test_hooks(self):
        """Test hook registration and activation inspection."""
        # Fit model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Register hook
        success = self.model.register_hook("user_encoder", None)
        self.assertTrue(success)
        
        # Get recommendations to trigger forward pass
        user_id = self.user_ids[0]
        self.model.recommend(user_id, top_n=5)
        
        # Check activations
        activation = self.model.hooks.get_activation("user_encoder")
        self.assertIsNotNone(activation)
        
        # Remove hook
        success = self.model.remove_hook("user_encoder")
        self.assertTrue(success)
        
        # Clear activations
        self.model.hooks.clear_activations()
        activation = self.model.hooks.get_activation("user_encoder")
        self.assertIsNone(activation)
    
    def test_incremental_update(self):
        """Test incremental model update."""
        # Fit model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Create new users and items
        new_users = [f'new_user_{i}' for i in range(3)]
        new_items = [f'new_item_{i}' for i in range(2)]
        
        # Create new interaction matrix with old and new users/items
        new_num_users = self.num_users + len(new_users)
        new_num_items = self.num_items + len(new_items)
        new_matrix = sp.lil_matrix((new_num_users, new_num_items))
        
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
    
    def test_export_embeddings(self):
        """Test exporting embeddings."""
        # Fit model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Export embeddings
        embeddings = self.model.export_embeddings()
        
        # Check that we have user and item embeddings
        self.assertIn('user_embeddings', embeddings)
        self.assertIn('item_embeddings', embeddings)
        
        # Check that all users and items have embeddings
        self.assertEqual(len(embeddings['user_embeddings']), self.num_users)
        self.assertEqual(len(embeddings['item_embeddings']), self.num_items)
        
        # Check embedding dimensions
        for uid, emb in embeddings['user_embeddings'].items():
            self.assertEqual(len(emb), self.latent_dim)
        
        for iid, emb in embeddings['item_embeddings'].items():
            self.assertEqual(len(emb), self.latent_dim)
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        # Fit model with seed 42
        model1 = BiVAE_base(
            name="TestBiVAE1",
            config={
                'latent_dim': self.latent_dim,
                'encoder_hidden_dims': [16, 8],
                'decoder_hidden_dims': [8, 16],
                'batch_size': 4,
                'num_epochs': 2,
                'device': 'cpu',
                'learning_rate': 0.01,
                'beta': 0.1,
                'seed': 42
            },
            seed=42
        )
        model1.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Fit another model with same seed
        model2 = BiVAE_base(
            name="TestBiVAE2",
            config={
                'latent_dim': self.latent_dim,
                'encoder_hidden_dims': [16, 8],
                'decoder_hidden_dims': [8, 16],
                'batch_size': 4,
                'num_epochs': 2,
                'device': 'cpu',
                'learning_rate': 0.01,
                'beta': 0.1,
                'seed': 42
            },
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


if __name__ == '__main__':
    unittest.main() 