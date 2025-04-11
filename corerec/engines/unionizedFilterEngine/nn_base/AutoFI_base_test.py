import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
import scipy.sparse as sp

from corerec.engines.unionizedFilterEngine.nn_base import AutoFI_base
from corerec.utils.hook_manager import HookManager

class TestAutoFIBase(unittest.TestCase):
    """Test the AutoFI_base class."""
    
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
        
        # Initialize the AutoFI model with small dimensions for testing
        self.model = AutoFI_base(
            name="TestAutoFI",
            trainable=True,
            verbose=True,
            config={
                'embedding_dim': 8,
                'hidden_dims': [16, 8],
                'num_interactions': 3,
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
        """Test initialization of AutoFI_base."""
        self.assertEqual(self.model.name, "TestAutoFI")
        self.assertTrue(self.model.trainable)
        self.assertTrue(self.model.verbose)
        self.assertEqual(self.model.config['embedding_dim'], 8)
        self.assertEqual(self.model.config['num_interactions'], 3)
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
        save_path = os.path.join(self.temp_dir, "autofi_model")
        self.model.save(save_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(f"{save_path}.pkl"))
        self.assertTrue(os.path.exists(f"{save_path}.meta"))
        
        # Load the model
        loaded_model = AutoFI_base.load(f"{save_path}.pkl")
        
        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.name, self.model.name)
        self.assertEqual(loaded_model.num_users, self.model.num_users)
        self.assertEqual(loaded_model.num_items, self.model.num_items)
        self.assertEqual(loaded_model.config['embedding_dim'], self.model.config['embedding_dim'])
        
        # Test recommendation with loaded model
        user_id = self.user_ids[0]
        recommendations = loaded_model.recommend(user_id, top_n=5)
        self.assertIsInstance(recommendations, list)
    
    def test_register_hook(self):
        """Test register_hook method."""
        # Build model first
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Register hook
        success = self.model.register_hook('embedding')
        self.assertTrue(success)
        
        # Make a prediction to trigger the hook
        user_id = self.user_ids[0]
        self.model.recommend(user_id, top_n=5)
        
        # Check activation
        activation = self.model.hooks.get_activation('embedding')
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
    
    def test_get_important_interactions(self):
        """Test get_important_interactions method."""
        # Fit the model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Get important interactions
        interactions = self.model.get_important_interactions(batch_size=10)
        
        # Check format
        self.assertEqual(len(interactions), self.model.config['num_interactions'])
        for interaction in interactions:
            self.assertIsInstance(interaction, tuple)
            field_i, field_j, importance = interaction
            self.assertIsInstance(field_i, int)
            self.assertIsInstance(field_j, int)
            self.assertIsInstance(importance, float)
            self.assertLess(field_i, field_j)  # Ensure field_i < field_j
    
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
        model1 = AutoFI_base(
            name="TestAutoFI1",
            config={'seed': 42, 'num_epochs': 2, 'device': 'cpu'},
            seed=42
        )
        model1.fit(self.interaction_matrix, self.user_ids, self.item_ids)
        
        # Fit another model with same seed
        model2 = AutoFI_base(
            name="TestAutoFI2",
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


if __name__ == '__main__':
    unittest.main() 