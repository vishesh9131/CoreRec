import pandas as pd
import numpy as np
from corerec.engines.collaborative.nn_base.DIEN_base import DIEN_base

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(num_users=100, num_items=50, num_interactions=1000):
    """Generate synthetic data for demonstration."""
    # Create users and items
    users = [f"user_{i}" for i in range(num_users)]
    items = [f"item_{i}" for i in range(num_items)]

    # Generate interactions
    interactions = []

    # Create sequences for each user
    for user in users:
        # Each user interacts with 5-15 items in sequence
        num_user_items = np.random.randint(5, 16)
        selected_items = np.random.choice(items, size=num_user_items, replace=False)

        # Add interactions with timestamps and features
        for i, item in enumerate(selected_items):
            features = {
                "category": np.random.choice(["electronics", "books", "clothing"]),
                "price": np.random.uniform(10, 200),
                "rating": np.random.randint(1, 6),
                "is_new": np.random.choice([True, False]),
                "discount": np.random.uniform(0, 0.5),
            }
            # Positive interaction
            interactions.append((user, item, features, 1))

    # Add some negative interactions
    for _ in range(num_interactions - len(interactions)):
        user = np.random.choice(users)
        item = np.random.choice(items)
        features = {
            "category": np.random.choice(["electronics", "books", "clothing"]),
            "price": np.random.uniform(10, 200),
            "rating": np.random.randint(1, 6),
            "is_new": np.random.choice([True, False]),
            "discount": np.random.uniform(0, 0.5),
        }
        # Negative interaction
        interactions.append((user, item, features, 0))

    return interactions, users, items


def main():
    """Main function to demonstrate DIEN model."""
    print("Generating synthetic data...")
    interactions, users, items = generate_synthetic_data()

    # Split data into train and test
    np.random.shuffle(interactions)
    split_idx = int(0.8 * len(interactions))
    train_interactions = interactions[:split_idx]
    test_interactions = interactions[split_idx:]

    print(f"Training data: {len(train_interactions)} interactions")
    print(f"Testing data: {len(test_interactions)} interactions")

    # Initialize DIEN model with complete configuration dictionary
    config = {
        "embed_dim": 32,
        "mlp_dims": [128, 64],
        "attention_dims": [64],
        "gru_hidden_dim": 32,
        "aux_loss_weight": 0.2,
        "dropout": 0.2,
        "batch_size": 128,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "max_seq_length": 10,
        "verbose": True,
    }

    model = DIEN_base(name="DIEN-Demo", config=config)

    # Train model
    print("Training DIEN model...")
    model.fit(train_interactions)

    # Evaluate model on test set
    print("Evaluating model...")
    correct = 0
    total = 0

    for user, item, features, label in test_interactions:
        try:
            pred = model.predict(user, item, features)
            pred_label = 1 if pred >= 0.5 else 0
            if pred_label == label:
                correct += 1
            total += 1
        except ValueError:
            # Skip if user or item not in training data
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate recommendations for a user
    test_user = users[0]
    print(f"\nTop 5 recommendations for user {test_user}:")
    recommendations = model.recommend(test_user, top_n=5)
    for i, (item, score) in enumerate(recommendations):
        print(f"{i+1}. Item: {item}, Score: {score:.4f}")

    # Get user and item embeddings
    user_embeddings = model.get_user_embeddings()
    item_embeddings = model.get_item_embeddings()

    print(f"\nUser embedding dimensions: {list(user_embeddings.values())[0].shape}")
    print(f"Item embedding dimensions: {list(item_embeddings.values())[0].shape}")

    # Save model
    model.save("dien_model.pt")
    print("Model saved to dien_model.pt")


if __name__ == "__main__":
    main()
