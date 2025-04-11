from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
import logging
from datetime import datetime
from corerec.engines.unionizedFilterEngine.nn_base.DCN_base import DCN_base


class DCN(DCN_base):
    """
    Deep & Cross Network (DCN) model.
    
    DCN combines a cross network for explicit feature interactions with a deep
    neural network for implicit feature interactions.
    
    Architecture:
    
    ┌─────────────────┐
    │ Feature Inputs  │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Embeddings    │
    └────────┬────────┘
             │
             ├─────────────────┐
             │                 │
    ┌────────▼────────┐ ┌─────▼─────────┐
    │  Cross Network  │ │  Deep Network │
    └────────┬────────┘ └─────┬─────────┘
             │                │
             └────────┬───────┘
                      │
             ┌────────▼────────┐
             │  Concatenation  │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │  Output Layer   │
             └────────┬────────┘
                      │
             ┌────────▼────────┐
             │    Prediction   │
             └─────────────────┘
    
    Args:
        name: Name of the model.
        embedding_dim: Dimension of feature embeddings.
        num_cross_layers: Number of cross layers.
        deep_layers: List of hidden layer dimensions for the deep network.
        dropout: Dropout rate for the deep network.
        activation: Activation function for the deep network.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        seed: Random seed for reproducibility.
        device: Device to run the model on ('cpu' or 'cuda').
        
    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """
    
    def __init__(
        self,
        name: str = "DCN",
        embedding_dim: int = 16,
        num_cross_layers: int = 3,
        deep_layers: List[int] = None,
        dropout: float = 0.2,
        activation: str = 'relu',
        batch_size: int = 256,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        seed: Optional[int] = None,
        device: str = None
    ):
        deep_layers = deep_layers or [64, 32, 16]
        
        super().__init__(
            name=name,
            embedding_dim=embedding_dim,
            num_cross_layers=num_cross_layers,
            deep_layers=deep_layers,
            dropout=dropout,
            activation=activation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            seed=seed,
            device=device
        )
    
    def get_similar_items(self, item_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get similar items based on feature embeddings.
        
        Args:
            item_id: ID of the item to find similar items for.
            top_n: Number of similar items to return.
            
        Returns:
            List of (item_id, similarity_score) tuples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if item_id not in self.item_map:
            self.logger.warning(f"Item {item_id} not in training data.")
            return []
        
        # Get item embedding
        item_idx = self.item_map[item_id]
        item_indices = torch.tensor([[item_idx]]).to(self.device)
        
        # Get all item embeddings
        all_item_indices = torch.tensor([[i] for i in self.reverse_item_map.keys()]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get embeddings from the embedding layer
            item_embedding = self.model.embeddings(item_indices).reshape(-1)
            all_embeddings = self.model.embeddings(all_item_indices).reshape(len(self.item_map), -1)
        
        # Compute cosine similarity
        item_embedding = item_embedding.cpu().numpy()
        all_embeddings = all_embeddings.cpu().numpy()
        
        similarities = []
        for i, embedding in enumerate(all_embeddings):
            if i == item_idx:
                continue
                
            # Compute cosine similarity
            similarity = np.dot(item_embedding, embedding) / (
                np.linalg.norm(item_embedding) * np.linalg.norm(embedding)
            )
            
            # Get item ID
            similar_item_id = self.reverse_item_map[i]
            similarities.append((similar_item_id, float(similarity)))
        
        # Sort by similarity and get top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def get_related_features(self, feature_name: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get related features based on cross-layer weights.
        
        Args:
            feature_name: Name of the feature to find related features for.
            top_n: Number of related features to return.
            
        Returns:
            List of (feature_name, relatedness_score) tuples.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        if feature_name not in self.feature_names:
            self.logger.warning(f"Feature {feature_name} not in training data.")
            return []
        
        # Get feature index
        feature_idx = self.feature_names.index(feature_name)
        
        # Get cross layer weights
        cross_weights = []
        for i in range(self.num_cross_layers):
            layer_name = f'cross_net.cross_layers.{i}'
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # Get weights
                    weights = module.weight.cpu().detach().numpy()
                    cross_weights.append(weights)
        
        # Average weights across layers
        avg_weights = np.mean(np.concatenate(cross_weights, axis=1), axis=1)
        
        # Get weights for the target feature
        target_weights = avg_weights[feature_idx]
        
        # Compute relatedness to other features
        relatedness = []
        for i, name in enumerate(self.feature_names):
            if i == feature_idx:
                continue
                
            # Compute relatedness (using absolute correlation)
            score = abs(avg_weights[i] * target_weights)
            relatedness.append((name, float(score)))
        
        # Sort by relatedness and get top N
        relatedness.sort(key=lambda x: x[1], reverse=True)
        return relatedness[:top_n]
    
    def save_metadata(self, path: str) -> None:
        """
        Save model metadata.
        
        Args:
            path: Path to save metadata to.
            
        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        metadata = {
            'name': self.name,
            'type': 'DCN',
            'version': '1.0',
            'created_at': str(datetime.now()),
            'config': {
                'embedding_dim': self.embedding_dim,
                'num_cross_layers': self.num_cross_layers,
                'deep_layers': self.deep_layers,
                'dropout': self.dropout,
                'activation': self.activation,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs
            },
            'num_features': len(self.feature_names),
            'num_users': len(self.user_map),
            'num_items': len(self.item_map),
            'feature_names': self.feature_names,
            'trainable': True,
            'seed': self.seed
        }
        
        with open(f"{path}_metadata.yaml", 'w') as f:
            import yaml
            yaml.dump(metadata, f)


if __name__ == "__main__":
    # Example usage
    # Create some sample data
    data = [
        ('user1', 'item1', {'category': 'electronics', 'price': 100, 'new': True}),
        ('user1', 'item2', {'category': 'books', 'price': 20, 'new': False}),
        ('user2', 'item1', {'category': 'electronics', 'price': 100, 'new': True}),
        ('user2', 'item3', {'category': 'clothing', 'price': 50, 'new': True}),
        ('user3', 'item2', {'category': 'books', 'price': 20, 'new': False}),
        ('user3', 'item3', {'category': 'clothing', 'price': 50, 'new': True}),
    ]
    
    # Create model
    model = DCN(
        name="ExampleDCN",
        embedding_dim=8,
        num_cross_layers=2,
        deep_layers=[16, 8],
        num_epochs=5
    )
    
    # Train model
    model.fit(data)
    
    # Get recommendations
    recs = model.recommend('user1', top_n=2)
    print("Recommendations for user1:", recs)
    
    # Get similar items
    sims = model.get_similar_items('item1', top_n=2)
    print("Items similar to item1:", sims)
    
    # Get feature importance
    importance = model.export_feature_importance()
    print("Feature importance:", importance)
