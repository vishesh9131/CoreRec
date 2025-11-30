from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import logging
from datetime import datetime
from corerec.engines.unionizedFilterEngine.nn_base.caser_base import Caser_base


class Caser(Caser_base):
    """
    Convolutional Sequence Embedding Recommendation (Caser) model.

    Architecture:

    Input Sequence      User ID
         |                |
         v                v
    [Item Embedding]  [User Embedding]
         |                |
         v                v
    +-------------------+ |
    | Horizontal Conv   | |
    +-------------------+ |
               |          |
               v          |
    +-------------------+ |
    | Vertical Conv     | |
    +-------------------+ |
               |          |
               v          v
          [Concatenate]
               |
               v
        [MLP + Dropout]
               |
               v
         [Output Layer]
               |
               v
          [Prediction]

    This model uses horizontal and vertical convolutional filters to capture sequential patterns
    in user behavior sequences for next-item recommendation.

    Args:
        name: Name of the model.
        config: Configuration dictionary with the following keys:
            - embedding_dim: Dimension of item embeddings (default: 64)
            - num_h_filters: Number of horizontal convolutional filters (default: 16)
            - num_v_filters: Number of vertical convolutional filters (default: 4)
            - max_seq_len: Maximum sequence length to consider (default: 5)
            - dropout_rate: Dropout rate for MLP layers (default: 0.5)
            - learning_rate: Learning rate for optimizer (default: 0.001)
            - batch_size: Training batch size (default: 64)
            - num_epochs: Number of training epochs (default: 20)
            - negative_samples: Number of negative samples per positive (default: 3)
            - device: Device to run the model on ('cpu' or 'cuda')
        trainable: Whether the model should be trainable or frozen.
        verbose: Whether to show progress bars during training.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "Caser",
        config: Optional[Dict[str, Any]] = None,
        trainable: bool = True,
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        default_config = {
            "embedding_dim": 64,
            "num_h_filters": 16,
            "num_v_filters": 4,
            "max_seq_len": 5,
            "dropout_rate": 0.5,
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_epochs": 20,
            "negative_samples": 3,
            "device": "cpu",
        }

        # Override defaults with provided config
        if config:
            default_config.update(config)

        super().__init__(
            name=name, config=default_config, trainable=trainable, verbose=verbose, seed=seed
        )

        # Set up logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def recommend_for_user_interaction_history(
        self, user_history: List[str], top_n: int = 10, exclude_seen: bool = True, **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations based on a user's interaction history.

        Args:
            user_history: List of item IDs the user has interacted with, ordered chronologically.
            top_n: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has interacted with.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            List of (item_id, score) tuples, sorted by score in descending order.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Map item IDs to indices
        item_indices = []
        for item_id in user_history:
            if item_id in self.iid_map:
                item_indices.append(self.iid_map[item_id])
            else:
                self.logger.warning(f"Item ID {item_id} not found in training data, skipping.")

        # If we don't have any valid items, return empty recommendations
        if not item_indices:
            return []

        # Prepare sequence
        seq = item_indices[-self.config["max_seq_len"] :]
        if len(seq) < self.config["max_seq_len"]:
            # Pad with zeros
            seq = [0] * (self.config["max_seq_len"] - len(seq)) + seq

        # Convert to tensor
        seq_tensor = torch.LongTensor([seq]).to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            scores = self.model(seq_tensor).cpu().numpy()[0]

        # Create item ID to score mapping
        item_scores = []
        for i, score in enumerate(scores):
            # Skip padding item (index 0)
            if i == 0:
                continue

            # Map index to item ID
            if i - 1 < len(self.item_ids):
                item_id = self.item_ids[i - 1]

                # Exclude seen items if requested
                if exclude_seen and item_id in user_history:
                    continue

                item_scores.append((item_id, float(score)))

        # Sort by score and return top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:top_n]

    def recommend_for_new_user(
        self, items: List[str], top_n: int = 10, **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a new user based on a few items they like.

        Args:
            items: List of item IDs the user likes.
            top_n: Number of recommendations to generate.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            List of (item_id, score) tuples, sorted by score in descending order.
        """
        # For Caser, this is the same as recommend_for_user_interaction_history
        return self.recommend_for_user_interaction_history(
            user_history=items, top_n=top_n, exclude_seen=True, **kwargs
        )

    def get_similar_items(self, item_id: str, top_n: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """
        Get items similar to the given item.

        Args:
            item_id: Item ID to find similar items for.
            top_n: Number of similar items to return.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            List of (item_id, similarity) tuples, sorted by similarity in descending order.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Get item embedding
        if item_id not in self.iid_map:
            raise ValueError(f"Item ID {item_id} not found in training data.")

        item_idx = self.iid_map[item_id]

        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            item_emb = self.model.item_embedding(torch.LongTensor([item_idx]).to(self.device))
            all_embs = self.model.item_embedding(
                torch.arange(1, self.num_items + 1).to(self.device)
            )

        # Compute similarities
        item_emb = item_emb.cpu().numpy()
        all_embs = all_embs.cpu().numpy()

        # Compute cosine similarity
        similarities = np.dot(all_embs, item_emb.T).squeeze()

        # Create item ID to similarity mapping
        item_sims = []
        for i, sim in enumerate(similarities):
            # Skip the query item
            if i == item_idx - 1:  # -1 because item indices start at 1
                continue

            # Map index to item ID
            item_id_sim = self.item_ids[i]
            item_sims.append((item_id_sim, float(sim)))

        # Sort by similarity and return top N
        item_sims.sort(key=lambda x: x[1], reverse=True)
        return item_sims[:top_n]

    def save_metadata(self, path: str):
        """
        Save model metadata.

        Args:
            path: Path to save metadata to.
        """
        metadata = {
            "name": self.name,
            "type": "Caser",
            "version": "1.0",
            "created_at": str(datetime.now()),
            "config": self.config,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "trainable": self.trainable,
            "seed": self.seed,
        }

        with open(f"{path}.meta", "w") as f:
            import yaml

            yaml.dump(metadata, f)


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create some sample data
    data = [
        ("user1", "item1", 1),
        ("user1", "item2", 2),
        ("user1", "item3", 3),
        ("user2", "item1", 1),
        ("user2", "item3", 2),
        ("user2", "item4", 3),
        ("user3", "item2", 1),
        ("user3", "item3", 2),
        ("user3", "item5", 3),
    ]

    # Create model
    model = Caser(
        name="ExampleCaser",
        config={
            "embedding_dim": 32,
            "num_h_filters": 8,
            "num_v_filters": 4,
            "max_seq_len": 3,
            "num_epochs": 5,
            "batch_size": 2,
        },
    )

    # Train model
    model.fit(data)

    # Get recommendations
    recs = model.recommend("user1", top_n=3)
    print("Recommendations for user1:", recs)

    # Get similar items
    sims = model.get_similar_items("item1", top_n=3)
    print("Items similar to item1:", sims)
