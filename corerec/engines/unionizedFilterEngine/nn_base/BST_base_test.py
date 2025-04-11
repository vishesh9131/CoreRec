import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from corerec.engines.unionizedFilterEngine.nn_base.BST_base import (
    BST_base, BSTModel, MultiHeadAttention, 
    TransformerBlock, TokenEmbedding, PositionalEmbedding, HookManager
)


class TestBSTComponents(unittest.TestCase):
    """Test individual components of the BST model."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.seq_len = 10
        self.vocab_size = 100
        self.hidden_dim = 16
        self.num_heads = 2
        self.field_dims = [5, 10, 8]  # Feature field dimensions
    
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
    
    def test_feature_embedding(self):
        """Test feature embeddings."""
        # Create feature embedding
        feature_embeddings = nn.ModuleList([
            nn.Embedding(dim, self.hidden_dim) for dim in self.field_dims
        ])
        
        # Generate random feature values
        feature_values = torch.randint(0, 5, (self.batch_size, len(self.field_dims)))
        
        # Embed features
        feature_embeds = []
        for i, embedding in enumerate(feature_embeddings):
            feature_embeds.append(embedding(feature_values[:, i]))
        
        # Stack feature embeddings
        feature_embed = torch.stack(feature_embeds, dim=1)
        
        # Check shape
        self.assertEqual(feature_embed.shape, (self.batch_size, len(self.field_dims), self.hidden_dim))
    
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
        transformer = TransformerBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            feed_forward_dim=self.hidden_dim * 4,
            dropout=0.1
        )
        x = torch.rand((self.batch_size, self.seq_len, self.hidden_dim))
        mask = torch.ones((self.batch_size, self.seq_len, self.seq_len), dtype=torch.bool)
        output, attention_weights = transformer(x, mask)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in transformer.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_bst_model(self):
        """Test the BSTModel."""
        model = BSTModel(
            num_items=self.vocab_size,
            hidden_dim=self.hidden_dim,
            feature_field_dims=self.field_dims,
            num_layers=2,
            num_heads=self.num_heads,
            max_seq_len=self.seq_len,
            dropout=0.1
        )
        
        # Create input tensors
        seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        item = torch.randint(0, self.vocab_size, (self.batch_size,))
        features = torch.randint(0, 5, (self.batch_size, len(self.field_dims)))
        
        # Forward pass
        output, attention_weights = model(seq, item, features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check attention weights shape (list of tensors)
        self.assertEqual(len(attention_weights), 2)  # 2 transformer blocks
        self.assertEqual(attention_weights[0].shape, 
                         (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check that output is differentiable
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
    
    def test_get_attention_weights(self):
        """Test getting attention weights from the model."""
        model = BSTModel(
            num_items=self.vocab_size,
            hidden_dim=self.hidden_dim,
            feature_field_dims=self.field_dims,
            num_layers=2,
            num_heads=self.num_heads,
            max_seq_len=self.seq_len,
            dropout=0.1
        )
        
        # Create input tensors
        seq = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        item = torch.randint(0, self.vocab_size, (self.batch_size,))
        features = torch.randint(0, 5, (self.batch_size, len(self.field_dims)))
        
        # Get attention weights
        attention_weights = model.get_attention_weights(seq, item, features)
        
        # Check attention weights shape (list of tensors)
        self.assertEqual(len(attention_weights), 2)  # 2 transformer blocks
        self.assertEqual(attention_weights[0].shape, 
                         (self.batch_size, self.num_heads, self.seq_len, self.seq_len))


class TestBSTBase(unittest.TestCase):
    """Test the BST_base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test data
        self.num_users = 10
        self.num_items = 20
        self.user_ids = [f'user_{i}' for i in range(self.num_users)]
        self.item_ids = [f'item_{i}' for i in range(self.num_items)]
        
        # Create item features
        self.feature_fields = ['category', 'brand', 'price_range']
        self.item_features = {}
        for i, iid in enumerate(self.item_ids):
            self.item_features[iid] = {
                'category': i % 5,
                'brand': i % 10,
                'price_range': i % 3
            }
        
        # Create interaction data
        self.interactions = []
        for i, uid in enumerate(self.user_ids):
            # Each user interacts with 5 items
            for j in range(5):
                item_idx = (i + j) % self.num_items
                iid = self.item_ids[item_idx]
                # Add timestamp
                timestamp = 1000 + i * 10 + j
                self.interactions.append((uid, iid, timestamp))
        
        # Initialize the model
        self.model = BST_base(
            name="TestBST",
            config={
                'hidden_dim': 16,
                'num_heads': 2,
                'num_layers': 2,
                'max_seq_len': 5,
                'dropout': 0.1,
                'batch_size': 4,
                'num_epochs': 2,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'device': 'cpu',
                'early_stopping': True,
                'patience': 2
            }
        )
    
    def test_initialization(self):
        """Test model initialization."""
        # Check that model is initialized with correct config
        self.assertEqual(self.model.name, "TestBST")
        self.assertEqual(self.model.config['hidden_dim'], 16)
        self.assertEqual(self.model.config['num_heads'], 2)
        
        # Check that model is not fitted
        self.assertFalse(self.model.is_fitted)
        
        # Check hook manager
        self.assertIsInstance(self.model.hooks, HookManager)
    
    def test_process_data(self):
        """Test data processing."""
        # Process data
        self.model._process_data(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Check mappings
        self.assertEqual(len(self.model.uid_map), self.num_users)
        self.assertEqual(len(self.model.iid_map), self.num_items)
        
        # Check sequences
        self.assertEqual(len(self.model.user_sequences), self.num_users)
        
        # Check feature field dimensions
        self.assertEqual(len(self.model.feature_field_dims), len(self.feature_fields))
    
    def test_build_model(self):
        """Test model building."""
        # Process data
        self.model._process_data(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Build model
        self.model._build_model()
        
        # Check that model is built
        self.assertIsInstance(self.model.model, BSTModel)
        
        # Check model parameters
        self.assertEqual(self.model.model.hidden_dim, self.model.config['hidden_dim'])
        self.assertEqual(self.model.model.num_heads, self.model.config['num_heads'])
        self.assertEqual(self.model.model.num_layers, self.model.config['num_layers'])
        
        # Check feature field dimensions
        self.assertEqual(len(self.model.model.feature_field_dims), len(self.feature_fields))
    
    def test_fit(self):
        """Test model fitting."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Check that model is fitted
        self.assertTrue(self.model.is_fitted)
        
        # Check that optimizer and scheduler are created
        self.assertIsNotNone(getattr(self.model, 'optimizer', None))
    
    def test_predict(self):
        """Test prediction."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Make a prediction
        user_id = self.user_ids[0]
        item_id = self.item_ids[0]
        score = self.model.predict(user_id, item_id)
        
        # Check prediction
        self.assertIsInstance(score, float)
    
    def test_recommend(self):
        """Test recommendation."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Get recommendations
        user_id = self.user_ids[0]
        recommendations = self.model.recommend(user_id, top_n=5)
        
        # Check recommendations
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIsInstance(rec, tuple)
            self.assertEqual(len(rec), 2)
            self.assertIn(rec[0], self.item_ids)
            self.assertIsInstance(rec[1], float)
    
    def test_save_load(self):
        """Test saving and loading the model."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bst_model"
            
            # Save the model
            self.model.save(path)
            
            # Check that files are created
            self.assertTrue(os.path.exists(f"{path}.pkl"))
            self.assertTrue(os.path.exists(f"{path}.meta"))
            
            # Load the model
            loaded_model = BST_base.load(f"{path}.pkl")
            
            # Check that loaded model has the same attributes
            self.assertEqual(loaded_model.name, self.model.name)
            self.assertEqual(loaded_model.config, self.model.config)
            self.assertEqual(len(loaded_model.user_ids), len(self.model.user_ids))
            self.assertEqual(len(loaded_model.item_ids), len(self.model.item_ids))
            
            # Check that loaded model can make predictions
            user_id = self.user_ids[0]
            item_id = self.item_ids[0]
            score1 = self.model.predict(user_id, item_id)
            score2 = loaded_model.predict(user_id, item_id)
            
            # Scores should be identical since we're using the same weights
            self.assertAlmostEqual(score1, score2, places=5)
    
    def test_hooks(self):
        """Test model hooks."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Register a hook
        success = self.model.hooks.register_hook(self.model.model, 'transformer_blocks.0')
        self.assertTrue(success)
        
        # Make a prediction to trigger the hook
        user_id = self.user_ids[0]
        item_id = self.item_ids[0]
        self.model.predict(user_id, item_id)
        
        # Check that activation is stored
        activation = self.model.hooks.get_activation('transformer_blocks.0')
        self.assertIsNotNone(activation)
        
        # Remove hook
        success = self.model.hooks.remove_hook('transformer_blocks.0')
        self.assertTrue(success)
        
        # Clear activations
        self.model.hooks.clear_activations()
        activation = self.model.hooks.get_activation('transformer_blocks.0')
        self.assertIsNone(activation)
    
    def test_get_attention_weights(self):
        """Test getting attention weights."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Get attention weights
        user_id = self.user_ids[0]
        item_id = self.item_ids[0]
        attention_weights = self.model.get_attention_weights(user_id, item_id)
        
        # Check attention weights
        self.assertIsInstance(attention_weights, list)
        self.assertEqual(len(attention_weights), self.model.config['num_layers'])
        
        # Check attention weights shape
        weights = attention_weights[0]
        self.assertEqual(weights.shape[0], 1)  # Batch size 1
        self.assertEqual(weights.shape[1], self.model.config['num_heads'])
        self.assertEqual(weights.shape[2], self.model.config['max_seq_len'])
        self.assertEqual(weights.shape[3], self.model.config['max_seq_len'])
    
    def test_incremental_update(self):
        """Test incremental update."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Create new interactions
        new_users = [f'new_user_{i}' for i in range(3)]
        new_items = [f'new_item_{i}' for i in range(2)]
        
        # Create new item features
        new_item_features = {}
        for i, iid in enumerate(new_items):
            new_item_features[iid] = {
                'category': i % 5,
                'brand': i % 10,
                'price_range': i % 3
            }
        
        # Create new interactions
        new_interactions = []
        
        # New users interact with existing items
        for i, uid in enumerate(new_users):
            for j in range(3):
                item_idx = (i + j) % self.num_items
                iid = self.item_ids[item_idx]
                timestamp = 2000 + i * 10 + j
                new_interactions.append((uid, iid, timestamp))
        
        # Existing users interact with new items
        for i, uid in enumerate(self.user_ids[:5]):
            for j, iid in enumerate(new_items):
                timestamp = 2000 + i * 10 + j + 100
                new_interactions.append((uid, iid, timestamp))
        
        # Update model incrementally
        self.model.update_incremental(
            new_interactions, 
            new_user_ids=new_users, 
            new_item_ids=new_items,
            new_item_features=new_item_features
        )
        
        # Check that model is updated
        self.assertEqual(len(self.model.user_ids), self.num_users + len(new_users))
        self.assertEqual(len(self.model.item_ids), self.num_items + len(new_items))
        
        # Check that we can make predictions for new users and items
        new_user_id = new_users[0]
        new_item_id = new_items[0]
        score = self.model.predict(new_user_id, new_item_id)
        self.assertIsInstance(score, float)
        
        # Get recommendations for new user
        recommendations = self.model.recommend(new_user_id, top_n=5)
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), 5)
    
    def test_export_embeddings(self):
        """Test exporting embeddings."""
        # Fit the model
        self.model.fit(self.interactions, self.user_ids, self.item_ids, self.item_features)
        
        # Export embeddings
        embeddings = self.model.export_embeddings()
        
        # Check embeddings
        self.assertIsInstance(embeddings, dict)
        self.assertEqual(len(embeddings), self.num_items)
        
        # Check embedding dimensions
        for iid, emb in embeddings.items():
            self.assertEqual(len(emb), self.model.config['hidden_dim'])


if __name__ == '__main__':
    unittest.main() 