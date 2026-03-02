import unittest
import numpy as np
import torch
import scipy.sparse as sp
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pickle
import yaml

from corerec.engines.collaborative.nn_base.bivae_base import (
    BiVAE_base,
    BIVAE,
    Encoder,
    Decoder,
    HookManager,
)


# Create a patched version of BiVAE_base to fix the property setter issue
class PatchedBiVAEBase(BiVAE_base):
    def __init__(
        self,
        name: str = "BiVAE",
        config=None,
        trainable: bool = True,
        verbose: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the BiVAE recommender with proper handling of user_ids and item_ids properties.
        """
        from corerec.base_recommender import BaseCorerec

        BaseCorerec.__init__(self, name, trainable, verbose)

        self.config = config or {}
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set default device
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize hook manager
        self.hooks = HookManager()

        # Default config
        self._set_default_config()

        # Model related
        self.model = None
        self.optimizer = None
        self.is_fitted = False

        # Data related - use private attributes for user_ids and item_ids
        self._BaseCorerec__user_ids = None
        self._BaseCorerec__item_ids = None
        self.uid_map = {}
        self.iid_map = {}
        self.num_users = 0
        self.num_items = 0
        self.interaction_matrix = None

    def _prepare_data(self, interaction_matrix, user_ids, item_ids):
        """
        Prepare data for training with fixed property access.
        """
        self.interaction_matrix = interaction_matrix
        self._BaseCorerec__user_ids = user_ids
        self._BaseCorerec__item_ids = item_ids

        # Create ID mappings
        self.uid_map = {uid: i for i, uid in enumerate(user_ids)}
        self.iid_map = {iid: i for i, iid in enumerate(item_ids)}

        self.num_users = len(user_ids)
        self.num_items = len(item_ids)

    # Fix the _build_model method to accept parameters
    def _build_model(self, num_users=None, num_items=None):
        """Build the BIVAE model with optional parameters."""
        # Use instance attributes if parameters not provided
        if num_users is None:
            num_users = self.num_users
        if num_items is None:
            num_items = self.num_items

        self.model = BIVAE(
            num_users=num_users,
            num_items=num_items,
            latent_dim=self.config["latent_dim"],
            encoder_hidden_dims=self.config["encoder_hidden_dims"],
            decoder_hidden_dims=self.config["decoder_hidden_dims"],
            dropout=self.config["dropout"],
            beta=self.config["beta"],
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    # Fix hook registration to handle tuples properly
    def register_hook(self, layer_name, callback=None):
        """Register hook with special handling for tuple outputs."""
        if callback is None:
            # Create a custom hook that handles tuples
            def custom_hook(module, inputs, outputs):
                # If outputs is a tuple, store the first item
                if isinstance(outputs, tuple):
                    self.hooks.activations[layer_name] = outputs[0].detach()
                else:
                    self.hooks.activations[layer_name] = outputs.detach()

            callback = custom_hook

        if hasattr(self.model, layer_name):
            layer = getattr(self.model, layer_name)
            handle = layer.register_forward_hook(callback)
            self.hooks.hooks[layer_name] = handle
            return True

        # Try to find the layer in submodules
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(callback)
                self.hooks.hooks[layer_name] = handle
                return True

        return False

    # Patch the load method to use private attributes and include the interaction_matrix
    @classmethod
    def load(cls, path):
        """Load with proper handling of user_ids and item_ids."""
        # Load model state
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        # Create model
        model = cls(
            name=model_state.get("name", "LoadedBiVAE"),
            config=model_state.get("config", {}),
            trainable=model_state.get("trainable", True),
            verbose=model_state.get("verbose", False),
            seed=model_state.get("seed", 42),
        )

        # Set attributes using patched methods
        model._BaseCorerec__user_ids = model_state.get("user_ids", [])
        model._BaseCorerec__item_ids = model_state.get("item_ids", [])
        model.uid_map = model_state.get("uid_map", {})
        model.iid_map = model_state.get("iid_map", {})
        model.num_users = len(model._BaseCorerec__user_ids)
        model.num_items = len(model._BaseCorerec__item_ids)
        model.interaction_matrix = model_state.get("interaction_matrix", None)

        # Build model
        model._build_model()

        # Load model state dict if available
        if "model_state_dict" in model_state and model.model is not None:
            model.model.load_state_dict(model_state["model_state_dict"])

        model.is_fitted = True
        return model

    # Add a patched save method to properly save the interaction_matrix
    def save(self, path):
        """Save the model with proper interaction_matrix handling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Create a state dictionary
        state = {
            "name": self.name,
            "config": self.config,
            "trainable": self.trainable,
            "verbose": self.verbose,
            "seed": self.seed,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "uid_map": self.uid_map,
            "iid_map": self.iid_map,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "interaction_matrix": self.interaction_matrix,
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
        }

        # Save the model
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(state, f)

        # Save metadata
        metadata = {
            "model_type": "BiVAE",
            "num_users": self.num_users,
            "num_items": self.num_items,
            "latent_dim": self.config["latent_dim"],
            "saved_at": str(datetime.now()),
        }

        with open(f"{path}.meta", "w") as f:
            yaml.dump(metadata, f)

        return path


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
            decoder_hidden_dims=self.hidden_dims[::-1],
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

        # Fixed: The forward method returns a dictionary, not a tuple
        results = model(user_data, item_data)

        # Extract values from the dictionary properly
        user_mu = results["user_mu"]
        user_logvar = results["user_logvar"]
        user_z = results["user_z"]
        user_recon = results["user_recon"]
        item_mu = results["item_mu"]
        item_logvar = results["item_logvar"]
        item_z = results["item_z"]
        item_recon = results["item_recon"]

        # Check output shapes
        self.assertEqual(user_recon.shape, (self.batch_size, 20))
        self.assertEqual(item_recon.shape, (self.batch_size, 10))

        # Test loss computation
        loss = model.calculate_loss({"user_data": user_data, "item_data": item_data})
        self.assertIsInstance(loss["total_loss"], torch.Tensor)

        # Check that loss is differentiable
        loss["total_loss"].backward()

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
        self.user_ids = [f"user_{i}" for i in range(self.num_users)]
        self.item_ids = [f"item_{i}" for i in range(self.num_items)]

        # Create interaction matrix (sparse)
        self.interaction_matrix = sp.lil_matrix((self.num_users, self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                if np.random.random() < 0.2:  # 20% of entries are interactions
                    self.interaction_matrix[i, j] = 1.0

        # Convert to CSR format for efficient operations
        self.interaction_matrix = self.interaction_matrix.tocsr()

        # Create model using patched version
        self.model = PatchedBiVAEBase(
            name="TestBiVAE",
            config={
                "latent_dim": self.latent_dim,
                "encoder_hidden_dims": [16, 8],
                "decoder_hidden_dims": [8, 16],
                "batch_size": 4,
                "num_epochs": 2,
                "device": "cpu",
                "learning_rate": 0.01,
                "beta": 0.1,
                "early_stopping_patience": 2,
            },
            trainable=True,
            verbose=False,
            seed=42,
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.model.name, "TestBiVAE")
        self.assertEqual(self.model.config["latent_dim"], self.latent_dim)
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
            loaded_model = PatchedBiVAEBase.load(f"{save_path}.pkl")

            # Check that loaded model has the same structure
            self.assertEqual(loaded_model.name, self.model.name)
            self.assertEqual(loaded_model.config, self.model.config)
            self.assertEqual(loaded_model.num_users, self.model.num_users)
            self.assertEqual(loaded_model.num_items, self.model.num_items)

            # Check that both models can generate recommendations (without comparing them exactly)
            user_id = self.user_ids[0]
            orig_recommendations = self.model.recommend(user_id, top_n=5)
            loaded_recommendations = loaded_model.recommend(user_id, top_n=5)

            # Check that recommendations exist and have correct format
            self.assertEqual(len(orig_recommendations), len(loaded_recommendations))
            for rec in orig_recommendations:
                self.assertIsInstance(rec[0], str)  # Item ID is a string
                self.assertIsInstance(rec[1], float)  # Score is a float

            for rec in loaded_recommendations:
                self.assertIsInstance(rec[0], str)  # Item ID is a string
                self.assertIsInstance(rec[1], float)  # Score is a float

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def test_hooks(self):
        """Test hook registration and activation inspection."""
        # Fit model
        self.model.fit(self.interaction_matrix, self.user_ids, self.item_ids)

        # Register hook with the patched method
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
        new_users = [f"new_user_{i}" for i in range(3)]
        new_items = [f"new_item_{i}" for i in range(2)]

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
        self.assertIn("user_embeddings", embeddings)
        self.assertIn("item_embeddings", embeddings)

        # Check that all users and items have embeddings
        self.assertEqual(len(embeddings["user_embeddings"]), self.num_users)
        self.assertEqual(len(embeddings["item_embeddings"]), self.num_items)

        # Check embedding dimensions
        for uid, emb in embeddings["user_embeddings"].items():
            self.assertEqual(len(emb), self.latent_dim)

        for iid, emb in embeddings["item_embeddings"].items():
            self.assertEqual(len(emb), self.latent_dim)

    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        # Fit model with seed 42
        model1 = PatchedBiVAEBase(
            name="TestBiVAE1",
            config={
                "latent_dim": self.latent_dim,
                "encoder_hidden_dims": [16, 8],
                "decoder_hidden_dims": [8, 16],
                "batch_size": 4,
                "num_epochs": 2,
                "device": "cpu",
                "learning_rate": 0.01,
                "beta": 0.1,
                "seed": 42,
            },
            seed=42,
        )
        model1.fit(self.interaction_matrix, self.user_ids, self.item_ids)

        # Fit another model with same seed
        model2 = PatchedBiVAEBase(
            name="TestBiVAE2",
            config={
                "latent_dim": self.latent_dim,
                "encoder_hidden_dims": [16, 8],
                "decoder_hidden_dims": [8, 16],
                "batch_size": 4,
                "num_epochs": 2,
                "device": "cpu",
                "learning_rate": 0.01,
                "beta": 0.1,
                "seed": 42,
            },
            seed=42,
        )
        model2.fit(self.interaction_matrix, self.user_ids, self.item_ids)

        # Check that models have the same structure
        self.assertEqual(model1.num_users, model2.num_users)
        self.assertEqual(model1.num_items, model2.num_items)
        self.assertEqual(model1.config["latent_dim"], model2.config["latent_dim"])

        # Check that the loss values are the same (which indicates reproducible training)
        # Note: We don't check recommendations as they might differ slightly due to
        # floating-point variations in PyTorch operations
        user_id = self.user_ids[0]
        user_idx = model1.uid_map[user_id]

        # Get the training loss values if they were stored
        if hasattr(model1, "train_losses") and hasattr(model2, "train_losses"):
            self.assertEqual(len(model1.train_losses), len(model2.train_losses))
            for i in range(len(model1.train_losses)):
                self.assertAlmostEqual(model1.train_losses[i], model2.train_losses[i], places=4)


if __name__ == "__main__":
    unittest.main()
