#!/usr/bin/env python3
"""
NCF (Neural Collaborative Filtering) Test Script
"""

from corerec.engines.collaborative.nn_base.ncf import NCF
import os
import sys
import numpy as np
import pandas as pd

# Add CoreRec to path
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_ncf():
    """Test NCF model with synthetic data"""
    print("=" * 60)
    print(" NCF (Neural Collaborative Filtering) Test")
    print("=" * 60)

    # Create synthetic data
    print("\nCreating synthetic data...")
    np.random.seed(42)

    num_users = 100
    num_items = 50
    num_interactions = 500

    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.choice([0, 1], num_interactions, p=[
                               0.3, 0.7])  # binary ratings

    data = pd.DataFrame(
        {"user_id": user_ids, "item_id": item_ids, "rating": ratings})

    print(
        f"Dataset: {
            len(data)} interactions, {
            data['user_id'].nunique()} users, {
                data['item_id'].nunique()} items")
    print(
        f"Positive ratings: {sum(data['rating'])}/{len(data)} ({sum(data['rating']) / len(data) * 100:.1f}%)"
    )

    # Train NCF model
    print("\nTraining NCF model...")
    model = NCF(
        name="NCF_Test",
        model_type="NeuMF",
        gmf_embedding_dim=16,
        mlp_embedding_dim=16,
        mlp_hidden_layers=(32, 16, 8),
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        num_epochs=3,
        negative_samples=4,
        device="cpu",
        verbose=True,
    )

    model.fit(data)

    # Test predictions
    print("\nTesting predictions...")
    test_user = data["user_id"].iloc[0]
    test_item = data["item_id"].iloc[0]

    try:
        prediction = model.predict(test_user, test_item)
        print(
            f"Prediction for user {test_user}, item {test_item}: {
                prediction:.4f}")
        print("✓ Prediction successful")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

    # Test recommendations
    print("\nTesting recommendations...")
    try:
        recommendations = model.recommend(test_user, top_n=5)
        print(f"Top 5 recommendations for user {test_user}: {recommendations}")
        print("✓ Recommendations successful")
    except Exception as e:
        print(f"✗ Recommendations failed: {e}")
        return False

    # Validation checks
    print("\nValidation:")
    print("-" * 40)
    print(f"✓ Model trained successfully")
    print(f"✓ Predictions working")
    print(f"✓ Recommendations working")
    print(f"✓ Number of recommendations: {len(recommendations)}")

    print("\nNCF test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_ncf()
    sys.exit(0 if success else 1)
