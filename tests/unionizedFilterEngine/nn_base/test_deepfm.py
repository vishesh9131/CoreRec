#!/usr/bin/env python3
"""
DeepFM Test Script
"""

from corerec.engines.collaborative.nn_base.DeepFM_base import DeepFMModel
import os
import sys
import numpy as np
import pandas as pd
import torch

# Add CoreRec to path
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_deepfm():
    """Test DeepFM model with synthetic data"""
    print("=" * 60)
    print(" DeepFM (Deep Factorization Machine) Test")
    print("=" * 60)

    # Create synthetic data
    print("\nCreating synthetic data...")
    np.random.seed(42)

    num_samples = 1000
    num_features = 10

    # Create feature data
    X = np.random.randint(0, 100, (num_samples, num_features))
    y = np.random.choice([0, 1], num_samples, p=[0.4, 0.6])

    print(f"Dataset: {num_samples} samples, {num_features} features")
    print(
        f"Positive samples: {sum(y)}/{len(y)} ({sum(y) / len(y) * 100:.1f}%)")

    # Train DeepFM model
    print("\nTraining DeepFM model...")

    # Create field dimensions (vocab size for each feature)
    field_dims = [100] * num_features

    try:
        model = DeepFMModel(
            field_dims=field_dims,
            embed_dim=16,
            mlp_dims=[
                64,
                32,
                16],
            dropout=0.2)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Convert to tensors
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Simple training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/3, Loss: {loss.item():.4f}")

        print("✓ Model trained successfully")

        # Test prediction
        print("\nTesting predictions...")
        model.eval()
        with torch.no_grad():
            test_sample = X_tensor[:5]
            predictions = torch.sigmoid(model(test_sample))
            print(f"Sample predictions: {predictions.numpy()}")
            print("✓ Predictions successful")

    except Exception as e:
        print(f"✗ DeepFM test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\nDeepFM test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_deepfm()
    sys.exit(0 if success else 1)
