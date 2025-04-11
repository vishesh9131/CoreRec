import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import random

from corerec.engines.unionizedFilterEngine.nn_base.Bert4Rec_base import (
     BERT4RecModel, MultiHeadAttention, 
    TransformerBlock, TokenEmbedding, PositionalEmbedding, HookManager
)


class TestBERT4RecComponents(unittest.TestCase):
    """Test individual components of the BERT4Rec model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.seq_len = 10
        self.vocab_size = 100
        self.hidden_dim = 16
        self.num_heads = 2
    
    def test_token_embedding(self):
        """Test the TokenEmbedding module."""
        embedding = TokenEmbedding(self.vocab_size, self.hidden_dim)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = embedding(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(embedding.embedding.weight.grad)
    
    def test_positional_embedding(self):
        """Test the PositionalEmbedding module."""
        embedding = PositionalEmbedding(self.seq_len, self.hidden_dim)
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = embedding(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(embedding.embedding.weight.grad)
    
    def test_multi_head_attention(self):
        """Test the MultiHeadAttention module."""
        attention = MultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )
        x = torch.rand((self.batch_size, self.seq_len, self.hidden_dim))
        mask = torch.ones((self.batch_size, self.seq_len, self.seq_len), dtype=torch.bool)
        output, attention_weights = attention(x, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in attention.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_transformer_block(self):
        """Test the TransformerBlock module."""
        block = TransformerBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.hidden_dim * 4,
            dropout=0.1
        )
        x = torch.rand((self.batch_size, self.seq_len, self.hidden_dim))
        mask = torch.ones((self.batch_size, self.seq_len, self.seq_len), dtype=torch.bool)
        output, attention_weights = block(x, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_bert4rec_model(self):
        """Test the BERT4RecModel."""
        model = BERT4RecModel(
            vocab_size=self.vocab_size,
            max_seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=2,
            dropout=0.1
        )
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        logits, attention_weights = model(x)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))
        
        # Check attention weights
        self.assertEqual(len(attention_weights), 2)  # 2 layers
        self.assertEqual(attention_weights[0].shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check that output is differentiable
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_hook_manager(self):
        """Test the HookManager."""
        hook_manager = HookManager()
        
        # Create a simple model to test hooks
        model = nn.Sequential(
            nn.Linear(10, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 1, bias=True)
        )
        model[0].name = 'linear1'
        model[2].name = 'linear2'
        
        # Register hook
        success = hook_manager.register_hook(model, '0')
        self.assertTrue(success)
        
        # Forward pass
        x = torch.rand(2, 10)
        output = model(x)
        
        # Check activation
        activation = hook_manager.get_activation('0')
        self.assertIsNotNone(activation)
        self.assertEqual(activation.shape, (2, 5))
        
        # Remove hook
        success = hook_manager.remove_hook('0')
        self.assertTrue(success)
        
        # Clear activations
        hook_manager.clear_activations()
        self.assertEqual(len(hook_manager.activations), 0)


class TestBERT4RecBase(unittest.TestCase):
    """Test the BERT4Rec_base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Create temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.num_users = 20
        self.num_items = 30
        
        # Create user and item IDs
        self.user_ids = [f"user_{i}" for i in range(self.num_users)]
        self.item_ids = [f"item_{i}" for i in range(self.num_items)]
        
        # Create interaction data with timestamps
        self.interactions = []
        for u in range(self.num_users):
            # Each user interacts with 5-10 random items
            num_interactions = random.randint(5, 10)
            items = random.sample(range(self.num_items), num_interactions)
            
            # Add interactions with timestamps
            for i, item in enumerate(items):
                self.interactions.append((self.user_ids[u], self.item_ids[item], i))
        
        # Create model with small dimensions for testing
        self.model = BERT4Rec_base(
            name="TestBERT4Rec",
            config={
                'hidden_dim': 16,
                'num_heads': 2,
                'num_layers': 2,
                'max_seq_len': 5,
                'mask_prob': 0.2,
                'batch_size': 4,
                'num_epochs': 2,
                'patience': 1,
                'learning_rate': 0.001,
                'device': 'cpu'
            },
            verbose=False
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "TestBERT4Rec")
        self.assertEqual(self.model.config['hidden_dim'], 16)
        self.assertEqual(self.model.config['num_heads'], 2)
        self.assertEqual(self.model.config['max_seq_len'], 5)
        self.assertFalse(self.model.is_fitted)
    
    def test_fit(self):
        """Test model fitting."""
        history = self.model.fit(self.interactions)
        
        # Check that model is fitted
        self.assertTrue(self.model.is_fitted)
        
        # Check that user and item mappings are created
        self.assertEqual(len(self.model.uid_map), self.num_users)
        self.assertEqual(len(self.model.iid_map), self.num_items)
        
        # Check that user sequences are created
        self.assertEqual(len(self.model.user_sequences), self.num_users)
        
        # Check that history contains loss
        self.assertIn('loss', history)
        
        # Check that loss decreases
        self.assertGreater(history['loss'][0], history['loss'][-1])
    
    def test_predict(self):
        """Test model prediction."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Choose a user-item pair that exists in the training data
        user_id = self.user_ids[0]
        item_id = self.item_ids[0]
        
        # Get prediction
        score = self.model.predict(user_id, item_id)
        
        # Check that score is a float
        self.assertIsInstance(score, float)
    
    def test_recommend(self):
        """Test model recommendation."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Choose a user
        user_id = self.user_ids[0]
        
        # Get recommendations
        recommendations = self.model.recommend(user_id, top_n=5)
        
        # Check that recommendations are returned
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        # Check format of recommendations
        for item_id, score in recommendations:
            self.assertIsInstance(item_id, str)
            self.assertIsInstance(score, float)
    
    def test_save_and_load(self):
        """Test model saving and loading."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Get recommendations before saving
        user_id = self.user_ids[0]
        original_recommendations = self.model.recommend(user_id, top_n=5)
        
        # Save the model
        save_path = os.path.join(self.temp_dir, "bert4rec_model")
        self.model.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{save_path}.pkl"))
        self.assertTrue(os.path.exists(f"{save_path}.meta"))
        
        # Load the model
        loaded_model = BERT4Rec_base.load(f"{save_path}.pkl")
        
        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.name, self.model.name)
        self.assertEqual(loaded_model.config['hidden_dim'], self.model.config['hidden_dim'])
        self.assertEqual(loaded_model.num_users, self.model.num_users)
        self.assertEqual(loaded_model.num_items, self.model.num_items)
        
        # Get recommendations after loading
        loaded_recommendations = loaded_model.recommend(user_id, top_n=5)
        
        # Check that recommendations are the same
        self.assertEqual(len(original_recommendations), len(loaded_recommendations))
        for i in range(len(original_recommendations)):
            self.assertEqual(original_recommendations[i][0], loaded_recommendations[i][0])
            self.assertAlmostEqual(original_recommendations[i][1], loaded_recommendations[i][1], places=5)
    
    def test_attention_weights(self):
        """Test getting attention weights."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Choose a user
        user_id = self.user_ids[0]
        
        # Get attention weights
        weights = self.model.get_attention_weights(user_id)
        
        # Check that weights are returned for each layer
        self.assertEqual(len(weights), self.model.config['num_layers'])
        
        # Check shape of weights
        max_seq_len = self.model.config['max_seq_len']
        num_heads = self.model.config['num_heads']
        self.assertEqual(weights[0].shape, (1, num_heads, max_seq_len, max_seq_len))
    
    def test_update_incremental(self):
        """Test incremental update."""
        # Fit the initial model
        self.model.fit(self.interactions)
        
        # Create new users and items
        new_users = [f"new_user_{i}" for i in range(3)]
        new_items = [f"new_item_{i}" for i in range(2)]
        
        # Create new interactions
        new_interactions = []
        
        # Existing users with existing items
        for u in range(5):
            for i in range(3):
                new_interactions.append((self.user_ids[u], self.item_ids[i], len(new_interactions)))
        
        # Existing users with new items
        for u in range(5):
            for i in range(len(new_items)):
                new_interactions.append((self.user_ids[u], new_items[i], len(new_interactions)))
        
        # New users with existing items
        for u in range(len(new_users)):
            for i in range(3):
                new_interactions.append((new_users[u], self.item_ids[i], len(new_interactions)))
        
        # New users with new items
        for u in range(len(new_users)):
            for i in range(len(new_items)):
                new_interactions.append((new_users[u], new_items[i], len(new_interactions)))
        
        # Update model incrementally
        updated_model = self.model.update_incremental(new_interactions, new_users, new_items)
        
        # Check that model was updated
        self.assertEqual(updated_model.num_users, self.num_users + len(new_users))
        self.assertEqual(updated_model.num_items, self.num_items + len(new_items))
        
        # Test recommendation for a new user
        recommendations = updated_model.recommend(new_users[0], top_n=5)
        self.assertIsInstance(recommendations, list)
    
    def test_export_embeddings(self):
        """Test exporting item embeddings."""
        # Fit the model
        self.model.fit(self.interactions)
        
        # Export embeddings
        embeddings = self.model.export_embeddings()
        
        # Check that embeddings are returned for all items
        self.assertEqual(len(embeddings), self.num_items)
        
        # Check embedding dimensions
        for item_id, emb in embeddings.items():
            self.assertEqual(len(emb), self.model.config['hidden_dim'])
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        # Fit model with seed 42
        model1 = BERT4Rec_base(
            name="TestBERT4Rec1",
            config={
                'hidden_dim': 16,
                'num_heads': 2,
                'num_layers': 2,
                'max_seq_len': 5,
                'mask_prob': 0.2,
                'batch_size': 4,
                'num_epochs': 2,
                'device': 'cpu',
                'seed': 42
            },
            seed=42
        )
        model1.fit(self.interactions)
        
        # Fit another model with same seed
        model2 = BERT4Rec_base(
            name="TestBERT4Rec2",
            config={
                'hidden_dim': 16,
                'num_heads': 2,
                'num_layers': 2,
                'max_seq_len': 5,
                'mask_prob': 0.2,
                'batch_size': 4,
                'num_epochs': 2,
                'device': 'cpu',
                'seed': 42
            },
            seed=42
        )
        model2.fit(self.interactions)
        
        # Get recommendations from both models
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