import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import torch.nn.functional as F

from corerec.base_recommender import BaseCorerec


class Generator(nn.Module):
    """
    Generator model for GAN-based Unified Filter.
    
    Takes noise vector concatenated with user features as input,
    and generates a item feature vector as output.
    """
    
    def __init__(self, layer_dims: List[int]):
        """
        Initialize the generator.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
            if i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(layer_dims[i+1]))
                layers.append(nn.LeakyReLU(0.2))
        
        # Final activation to match feature distribution
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            noise: Noise vector
            condition: User features (condition)
            
        Returns:
            Generated item features
        """
        batch_size = condition.size(0)
        if noise.size(0) != batch_size:
            # Resize noise tensor to match batch size if needed
            noise = noise[:batch_size]
        
        # Concatenate noise and user features
        x = torch.cat([noise, condition], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """
    Discriminator model for GAN-based Unified Filter.
    
    Takes user features and item features as input,
    and predicts whether the item is relevant to the user.
    """
    
    def __init__(self, layer_dims: List[int]):
        """
        Initialize the discriminator.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            
            if i < len(layer_dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.3))
        
        # Final activation for binary classification
        layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            user_features: User features
            item_features: Item features
            
        Returns:
            Probability that item is relevant to user
        """
        # Concatenate user and item features
        x = torch.cat([user_features, item_features], dim=1)
        return self.main(x)


class GAN_ufilter_base(BaseCorerec):
    """
    GAN-based Unified Filter recommendation model.
    
    Uses a generative adversarial network approach for recommendation,
    conditioning the generation on user features to create personalized recommendations.
    
    Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │                     GAN_ufilter                           │
    │                                                           │
    │  ┌─────────────┐    ┌───────────────┐    ┌────────────┐  │
    │  │ User & Item │    │Generator Model│    │Discriminator│  │
    │  └──────┬──────┘    └───────┬───────┘    └──────┬─────┘  │
    │         │                   │                   │         │
    │         └───────────┬───────┴─────────┬─────────┘         │
    │                     │                 │                    │
    │                     ▼                 ▼                    │
    │            ┌────────────────┐  ┌─────────────┐            │
    │            │ Adversarial    │  │Training Loop│            │
    │            │   Training     │  └──────┬──────┘            │
    │            └────────┬───────┘         │                    │
    │                     └────────┬────────┘                    │
    │                              │                             │
    │                              ▼                             │
    │                    ┌─────────────────┐                     │
    │                    │Recommendation API│                     │
    │                    └─────────────────┘                     │
    └───────────────────────────────────────────────────────────┘
    
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "GAN_UFilter",
        noise_dim: int = 64,
        gen_hidden_dims: List[int] = [128, 256],
        disc_hidden_dims: List[int] = [256, 128],
        learning_rate: float = 0.0002,
        batch_size: int = 64,
        num_epochs: int = 50,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = None,
        seed: int = 42,
        verbose: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the GAN-based recommendation model.
        
        Args:
            name: Model name
            noise_dim: Dimension of noise input for generator
            gen_hidden_dims: Hidden dimensions for generator
            disc_hidden_dims: Hidden dimensions for discriminator
            learning_rate: Learning rate for both generator and discriminator
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            device: Device to run model on ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            verbose: Whether to display training progress
            config: Configuration dictionary that overrides the default parameters
        """
        super().__init__(name=name, verbose=verbose)
        self.seed = seed
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Process config if provided
        if config is not None:
            self.noise_dim = config.get("noise_dim", noise_dim)
            self.gen_hidden_dims = config.get("gen_hidden_dims", gen_hidden_dims)
            self.disc_hidden_dims = config.get("disc_hidden_dims", disc_hidden_dims)
            self.learning_rate = config.get("learning_rate", learning_rate)
            self.batch_size = config.get("batch_size", batch_size)
            self.num_epochs = config.get("num_epochs", num_epochs)
            self.beta1 = config.get("beta1", beta1)
            self.beta2 = config.get("beta2", beta2)
        else:
            self.noise_dim = noise_dim
            self.gen_hidden_dims = gen_hidden_dims
            self.disc_hidden_dims = disc_hidden_dims
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.beta1 = beta1
            self.beta2 = beta2
            
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Setup logger
        self._setup_logger()
        
        # Initialize models and optimizers
        self.generator = None
        self.discriminator = None
        self.gen_optimizer = None
        self.disc_optimizer = None
        self.loss_fn = nn.BCELoss()
        
        # Initialize data structures
        self.user_features = None
        self.item_features = None
        self.user_encoding = None
        self.item_encoding = None
        self.num_users = 0
        self.num_items = 0
        self.user_embedding_dim = 0
        self.item_embedding_dim = 0
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} model")
    
    def _setup_logger(self):
        """Setup logger for the model."""
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def _preprocess_data(self, data: Dict[str, Any]):
        """
        Preprocess data for training.
        
        Args:
            data: Dictionary with user and item features, and interaction data
        """
        # Extract user and item features
        if 'user_features' in data and 'item_features' in data:
            self.user_features = data['user_features']
            self.item_features = data['item_features']
        elif 'users' in data and 'items' in data:
            # For compatibility with test data format
            self.user_features = data['users']
            self.item_features = data['items']
        else:
            raise ValueError("Data must contain user and item features")
        
        # Extract interactions
        if 'interactions' not in data:
            raise ValueError("Data must contain interactions")
        
        self.interactions = data['interactions']
        
        # Create user and item maps
        self.user_map = {user_id: i for i, user_id in enumerate(self.user_features.keys())}
        self.item_map = {item_id: i for i, item_id in enumerate(self.item_features.keys())}
        
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map)
        self.feature_dim = next(iter(self.user_features.values())).shape[0]
        
        # Convert features to tensors
        self.user_feature_tensor = torch.zeros((self.n_users, self.feature_dim), device=self.device)
        self.item_feature_tensor = torch.zeros((self.n_items, self.feature_dim), device=self.device)
        
        for user_id, features in self.user_features.items():
            user_idx = self.user_map[user_id]
            self.user_feature_tensor[user_idx] = torch.tensor(features, device=self.device)
            
        for item_id, features in self.item_features.items():
            item_idx = self.item_map[item_id]
            self.item_feature_tensor[item_idx] = torch.tensor(features, device=self.device)
            
        # Create positive samples
        self.positives = []
        for user_id, item_ids in self.interactions.items():
            if user_id not in self.user_map:
                continue
                
            user_idx = self.user_map[user_id]
            for item_id in item_ids:
                if item_id not in self.item_map:
                    continue
                item_idx = self.item_map[item_id]
                self.positives.append((user_idx, item_idx))
                
        self.positives = np.array(self.positives)
        
        if self.verbose:
            self.logger.info(f"Processed data: {self.n_users} users, {self.n_items} items")
            self.logger.info(f"Total interactions: {len(self.positives)}")
    
    def _build_models(self):
        """
        Build generator and discriminator models.
        """
        # Build generator
        generator_dims = [self.noise_dim + self.feature_dim] + self.gen_hidden_dims + [self.feature_dim]
        self.generator = Generator(generator_dims).to(self.device)
        
        # Build discriminator
        discriminator_dims = [self.feature_dim * 2] + self.disc_hidden_dims + [1]
        self.discriminator = Discriminator(discriminator_dims).to(self.device)
        
        # Setup optimizers
        self.gen_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        
        if self.verbose:
            self.logger.info(f"Generator architecture: {self.generator}")
            self.logger.info(f"Discriminator architecture: {self.discriminator}")
            
    def _sample_positive_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of positive user-item interactions.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Tuple of (user_indices, item_indices)
        """
        if len(self.positives) <= batch_size:
            indices = np.arange(len(self.positives))
        else:
            indices = np.random.choice(len(self.positives), batch_size, replace=False)
            
        batch = self.positives[indices]
        user_indices = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
        item_indices = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
        
        return user_indices, item_indices
        
    def _sample_negative_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of negative user-item interactions.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Tuple of (user_indices, item_indices)
        """
        user_indices = torch.randint(0, self.n_users, (batch_size,), device=self.device)
        item_indices = torch.randint(0, self.n_items, (batch_size,), device=self.device)
        
        return user_indices, item_indices
    
    def _get_user_features(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        Get user features from indices.
        
        Args:
            user_indices: User indices tensor
            
        Returns:
            User features tensor
        """
        return self.user_feature_tensor[user_indices]
    
    def _get_item_features(self, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Get item features from indices.
        
        Args:
            item_indices: Item indices tensor
            
        Returns:
            Item features tensor
        """
        return self.item_feature_tensor[item_indices]
    
    def predict(self, user_id: Any) -> np.ndarray:
        """
        Predict relevance scores for a user across all items.
        
        Args:
            user_id: User ID
            
        Returns:
            Array of recommendation scores for all items
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        # Get user features and convert to tensor
        if user_id not in self.user_map:
            raise ValueError(f"Unknown user ID: {user_id}")
            
        # Set to eval mode
        self.generator.eval()
        self.discriminator.eval()
        
        # Get user index and features
        user_idx = self.user_map[user_id]
        user_features = self.user_feature_tensor[user_idx].unsqueeze(0)
        
        # Generate multiple samples for the user
        n_samples = 10
        predictions = torch.zeros(n_samples, self.n_items, device=self.device)
        
        with torch.no_grad():
            for i in range(n_samples):
                # Generate noise
                noise = torch.randn(1, self.noise_dim, device=self.device)
                
                # Generate fake item features
                generated_item_features = self.generator(noise, user_features)
                
                # Calculate similarity to all real items (cosine similarity)
                item_features_norm = F.normalize(self.item_feature_tensor, dim=1)
                generated_norm = F.normalize(generated_item_features, dim=1)
                similarity = torch.matmul(generated_norm, item_features_norm.t()).squeeze(0)
                
                # Get scores through discriminator for all items
                for j in range(self.n_items):
                    item_features = self.item_feature_tensor[j].unsqueeze(0)
                    score = self.discriminator(user_features, item_features)
                    predictions[i, j] = score
        
        # Average scores across samples
        avg_predictions = predictions.mean(dim=0).cpu().numpy()
        
        return avg_predictions
    
    def recommend(self, user_id: Any, top_n: int = 10) -> List[Tuple[Any, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to generate
            
        Returns:
            List of (item_id, score) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
            
        # Get predictions for all items
        predictions = self.predict(user_id)
        
        # Create map of original item IDs to their index
        idx_to_item = {idx: item_id for item_id, idx in self.item_map.items()}
        
        # Get indices of top items based on predictions
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        # Create recommendation list
        recommendations = [(idx_to_item[idx], float(predictions[idx])) for idx in top_indices]
        
        return recommendations
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data to save
        model_data = {
            'model_config': {
                'name': self.name,
                'noise_dim': self.noise_dim,
                'gen_hidden_dims': self.gen_hidden_dims,
                'disc_hidden_dims': self.disc_hidden_dims,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'seed': self.seed,
                'verbose': self.verbose
            },
            'model_state': {
                'generator': self.generator.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'gen_optimizer': self.gen_optimizer.state_dict(),
                'disc_optimizer': self.disc_optimizer.state_dict()
            },
            'data': {
                'user_features': self.user_features,
                'item_features': self.item_features,
                'user_encoding': self.user_encoding,
                'item_encoding': self.item_encoding,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'user_embedding_dim': self.user_embedding_dim,
                'item_embedding_dim': self.item_embedding_dim
            },
            'history': {
                'disc_losses': self.disc_losses,
                'gen_losses': self.gen_losses
            }
        }
        
        # Save to file
        torch.save(model_data, filepath)
        
        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> 'GAN_ufilter_base':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        # Load model data
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        model_data = torch.load(filepath, map_location=device)
        
        # Create model instance with saved config
        config = model_data['model_config']
        instance = cls(
            name=config['name'],
            noise_dim=config['noise_dim'],
            gen_hidden_dims=config['gen_hidden_dims'],
            disc_hidden_dims=config['disc_hidden_dims'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            beta1=config['beta1'],
            beta2=config['beta2'],
            seed=config['seed'],
            verbose=config['verbose'],
            device=device
        )
        
        # Restore data
        data = model_data['data']
        instance.user_features = data['user_features']
        instance.item_features = data['item_features']
        instance.user_encoding = data['user_encoding']
        instance.item_encoding = data['item_encoding']
        instance.num_users = data['num_users']
        instance.num_items = data['num_items']
        instance.user_embedding_dim = data['user_embedding_dim']
        instance.item_embedding_dim = data['item_embedding_dim']
        
        # Restore history
        history = model_data['history']
        instance.disc_losses = history['disc_losses']
        instance.gen_losses = history['gen_losses']
        
        # Build and load models
        instance._build_models()
        
        # Load model states
        model_state = model_data['model_state']
        instance.generator.load_state_dict(model_state['generator'])
        instance.discriminator.load_state_dict(model_state['discriminator'])
        instance.gen_optimizer.load_state_dict(model_state['gen_optimizer'])
        instance.disc_optimizer.load_state_dict(model_state['disc_optimizer'])
        
        instance.is_fitted = True
        return instance
    
    def train(self):
        """Required by base class but implemented as fit."""
        pass

    def _train_step(self, batch_size: int) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        # Set models to train mode
        self.generator.train()
        self.discriminator.train()
        
        # Print debug info to understand dimensions
        print(f"Noise dim: {self.noise_dim}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Generator input dim: {self.generator.main[0].in_features}")
        
        # Train discriminator on real samples
        self.disc_optimizer.zero_grad()
        
        # Sample positive user-item pairs
        real_user_indices, real_item_indices = self._sample_positive_batch(batch_size)
        real_user_features = self._get_user_features(real_user_indices)
        real_item_features = self._get_item_features(real_item_indices)
        
        # Discriminator prediction on real samples
        d_real = self.discriminator(real_user_features, real_item_features)
        
        # Train discriminator on fake samples
        # Sample users
        fake_user_indices, _ = self._sample_positive_batch(batch_size)
        fake_user_features = self._get_user_features(fake_user_indices)
        
        # Ensure noise dimension matches first layer input size of generator minus feature size
        noise_dim = self.generator.main[0].in_features - self.feature_dim
        
        # Generate noise
        noise = torch.randn(batch_size, noise_dim, device=self.device)
        
        # Generate fake items
        fake_item_features = self.generator(noise, fake_user_features)
        
        # Discriminator prediction on fake samples
        d_fake = self.discriminator(fake_user_features, fake_item_features.detach())
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        d_loss_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        d_loss = d_loss_real + d_loss_fake
        
        # Update discriminator
        d_loss.backward()
        self.disc_optimizer.step()
        
        # Train generator
        self.gen_optimizer.zero_grad()
        
        # Sample new users
        gen_user_indices, _ = self._sample_positive_batch(batch_size)
        gen_user_features = self._get_user_features(gen_user_indices)
        
        # Generate new noise
        noise = torch.randn(batch_size, noise_dim, device=self.device)
        
        # Generate fake items
        fake_item_features = self.generator(noise, gen_user_features)
        
        # Discriminator prediction on generated samples
        d_gen = self.discriminator(gen_user_features, fake_item_features)
        
        # Generator loss
        g_loss = F.binary_cross_entropy(d_gen, torch.ones_like(d_gen))
        
        # Update generator
        g_loss.backward()
        self.gen_optimizer.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_real": d_real.mean().item(),
            "d_fake": d_fake.mean().item()
        }
    
    def fit(self, data: Dict[str, Any]) -> 'GAN_ufilter_base':
        """
        Train the model.
        
        Args:
            data: Training data
            
        Returns:
            Trained model instance
        """
        # Prepare data
        self._preprocess_data(data)
        
        # Build models
        self._build_models()
        
        if self.verbose:
            self.logger.info("Starting training...")
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_metrics = {
                "d_loss": 0.0,
                "g_loss": 0.0,
                "d_real": 0.0,
                "d_fake": 0.0
            }
            
            # Multiple batches per epoch
            num_batches = max(1, len(self.positives) // self.batch_size)
            for _ in range(num_batches):
                batch_metrics = self._train_step(self.batch_size)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v / num_batches
            
            if self.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"d_loss: {epoch_metrics['d_loss']:.4f}, "
                    f"g_loss: {epoch_metrics['g_loss']:.4f}, "
                    f"d_real: {epoch_metrics['d_real']:.4f}, "
                    f"d_fake: {epoch_metrics['d_fake']:.4f}"
                )
        
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info("Training completed")
            
        return self