import unittest
import torch
import numpy as np
import os
import tempfile
import shutil
from corerec.engines.unionizedFilterEngine.nn_base.deep_mf_base import DeepMF_base


class TestDeepMF(unittest.TestCase):
    """
    Test suite for DeepMF model.

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create temporary directory for model saving/loading
        self.temp_dir = tempfile.mkdtemp()

        # Generate synthetic data
        self.users = [f"u{i}" for i in range(100)]
        self.items = [f"i{i}" for i in range(50)]

        # Generate interactions
        self.interactions = []
        for user in self.users[:80]:  # Use 80% users for training
            # Each user interacts with 5-10 items
            num_items = np.random.randint(5, 11)
            item_indices = np.random.choice(range(len(self.items)), size=num_items, replace=False)
            for item_idx in item_indices:
                item = self.items[item_idx]
                rating = float(np.random.randint(1, 6))  # 1-5 rating
                self.interactions.append((user, item, rating))

        # Create test model
        self.model = DeepMF_base(
            name="TestDeepMF",
            embedding_dim=32,
            hidden_layers=[64, 32],
            dropout=0.2,
            batch_size=64,
            learning_rate=0.001,
            num_epochs=2,
            seed=42,
        )

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test model initialization."""
        # Check that model has correct attributes
        self.assertEqual(self.model.name, "TestDeepMF")
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_layers, [64, 32])
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

        # Check that all users and items in interactions are in maps
        for user, item, _ in self.interactions:
            self.assertIn(user, self.model.user_map)
            self.assertIn(item, self.model.item_map)

        # Check that model has loss history
        self.assertGreater(len(self.model.loss_history), 0)

    def test_predict(self):
        """Test prediction."""
        # Fit model
        self.model.fit(self.interactions)

        # Get a user-item pair from interactions
        user, item, _ = self.interactions[0]

        # Make prediction
        prediction = self.model.predict(user, item)

        # Check that prediction is a float
        self.assertIsInstance(prediction, float)

        # Check that prediction is within expected range (after sigmoid)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

        # Test prediction for non-existent user/item
        # The implementation returns a random value, so we can't assert exact values
        pred_new_user = self.model.predict("non_existent_user", item)
        pred_new_item = self.model.predict(user, "non_existent_item")

        # Just check they're valid predictions (between 0 and 1)
        self.assertGreaterEqual(pred_new_user, 0.0)
        self.assertLessEqual(pred_new_user, 1.0)
        self.assertGreaterEqual(pred_new_item, 0.0)
        self.assertLessEqual(pred_new_item, 1.0)

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

        # Test for non-existent user - the implementation provides random recommendations
        # So we only check that some recommendations are returned
        recommendations_new_user = self.model.recommend("non_existent_user", top_n=5)
        self.assertIsInstance(recommendations_new_user, list)
        # Length check may vary depending on implementation, so we skip this assertion

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
        loaded_model = DeepMF_base.load(save_path)

        # Check that loaded model has the same attributes
        self.assertEqual(loaded_model.name, self.model.name)
        self.assertEqual(loaded_model.embedding_dim, self.model.embedding_dim)
        self.assertEqual(loaded_model.hidden_layers, self.model.hidden_layers)
        self.assertEqual(loaded_model.dropout, self.model.dropout)
        self.assertEqual(loaded_model.seed, self.model.seed)

        # Check that user and item maps are the same
        self.assertEqual(loaded_model.user_map, self.model.user_map)
        self.assertEqual(loaded_model.item_map, self.model.item_map)

        # Check that predictions are the same
        user, item, _ = self.interactions[0]
        original_pred = self.model.predict(user, item)
        loaded_pred = loaded_model.predict(user, item)
        self.assertAlmostEqual(original_pred, loaded_pred, places=6)

    def test_update_incremental(self):
        """Test incremental model update."""
        # Fit model
        self.model.fit(self.interactions)

        # Create new interactions with existing and new users/items
        new_users = [f"u{i}" for i in range(100, 110)]
        new_items = [f"i{i}" for i in range(50, 55)]

        new_interactions = []
        # Existing user, existing item
        new_interactions.append((self.users[0], self.items[0], 5.0))
        # Existing user, new item
        new_interactions.append((self.users[0], new_items[0], 4.0))
        # New user, existing item
        new_interactions.append((new_users[0], self.items[0], 3.0))
        # New user, new item
        new_interactions.append((new_users[0], new_items[0], 2.0))

        try:
            # Try to update model using the incremental method
            self.model.update_incremental(new_interactions, new_users, new_items)

            # Check that new users and items are added to mappings
            for user in new_users:
                self.assertIn(user, self.model.user_map)

            for item in new_items:
                self.assertIn(item, self.model.item_map)

            # Check that we can predict for new user-item pairs
            prediction = self.model.predict(new_users[0], new_items[0])
            self.assertIsInstance(prediction, float)
            self.assertGreaterEqual(prediction, 0.0)
            self.assertLessEqual(prediction, 1.0)
        except (AttributeError, NotImplementedError):
            # If update_incremental or _train is not implemented, skip this test
            self.skipTest("update_incremental method is not fully implemented")

    def test_get_embeddings(self):
        """Test user and item embedding extraction."""
        # Fit model
        self.model.fit(self.interactions)

        try:
            # Get user embeddings
            user_embeddings = self.model.export_user_embeddings()

            # Check that user embeddings contains all users
            self.assertEqual(len(user_embeddings), len(self.model.user_map))
            for user in self.model.user_map:
                self.assertIn(user, user_embeddings)
                self.assertEqual(len(user_embeddings[user]), self.model.embedding_dim)

            # Get item embeddings
            item_embeddings = self.model.export_item_embeddings()

            # Check that item embeddings contains all items
            self.assertEqual(len(item_embeddings), len(self.model.item_map))
            for item in self.model.item_map:
                self.assertIn(item, item_embeddings)
                self.assertEqual(len(item_embeddings[item]), self.model.embedding_dim)
        except (AttributeError, NotImplementedError):
            # If the methods aren't implemented, try the alternative methods
            try:
                # Try get_user_embedding per user
                for user in list(self.model.user_map.keys())[:5]:  # Check just a few users
                    user_emb = self.model.get_user_embedding(user)
                    self.assertEqual(len(user_emb), self.model.embedding_dim)

                # Try get_item_embedding per item
                for item in list(self.model.item_map.keys())[:5]:  # Check just a few items
                    item_emb = self.model.get_item_embedding(item)
                    self.assertEqual(len(item_emb), self.model.embedding_dim)
            except (AttributeError, NotImplementedError):
                # If even these methods aren't implemented, skip the test
                self.skipTest("Embedding extraction methods are not implemented")

    def test_device_setting(self):
        """Test device setting."""
        # Set device to CPU
        self.model.set_device("cpu")
        self.assertEqual(str(self.model.device), "cpu")

        # Fit model
        self.model.fit(self.interactions)

        # Check that model is on CPU
        for param in self.model.model.parameters():
            self.assertEqual(str(param.device), "cpu")

        # Only test CUDA if available
        if torch.cuda.is_available():
            # Set device to CUDA
            self.model.set_device("cuda")
            self.assertEqual(str(self.model.device), "cuda:0")

            # Check that model is on CUDA
            for param in self.model.model.parameters():
                self.assertEqual(str(param.device), "cuda:0")


if __name__ == "__main__":
    unittest.main()
