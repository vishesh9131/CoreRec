#!/usr/bin/env python3
"""
GRU-CF (GRU-based Collaborative Filtering) Test Script
"""

from corerec.engines.unionizedFilterEngine.nn_base.gru_cf import GRUNet
import os
import sys
import numpy as np
import torch

# Add CoreRec to path
ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_gru_cf():
    """Test GRU4Rec model with synthetic sequential data"""
    print("=" * 60)
    print(" GRU4Rec (GRU for Recommendation) Test")
    print("=" * 60)

    # Create synthetic sequential data
    print("\nCreating synthetic sequential data...")
    np.random.seed(42)

    num_items = 100
    batch_size = 32
    seq_length = 10

    print(
        f"Dataset: batch_size={batch_size}, num_items={num_items}, seq_length={seq_length}")

    try:
        # Initialize GRU4Rec model
        print("\nInitializing GRU4Rec model...")
        model = GRUNet(
            num_items=num_items,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            dropout=0.2)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Create synthetic batch (item indices for embedding)
        sequences = torch.randint(0, num_items, (batch_size, seq_length))
        targets = torch.randint(0, num_items, (batch_size,))

        # Simple training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            # Model returns tuple (logits, hidden)
            logits, hidden = model(sequences)
            # Take the last timestep output
            outputs = logits[:, -1, :]  # (batch_size, num_items+1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/3, Loss: {loss.item():.4f}")

        print("✓ Model trained successfully")

        # Test prediction
        print("\nTesting predictions...")
        model.eval()
        with torch.no_grad():
            test_seqs = sequences[:5]
            logits, _ = model(test_seqs)
            predictions = torch.softmax(logits[:, -1, :], dim=1)
            top_items = torch.topk(predictions, k=5, dim=1)[1]
            print(
                f"Top-5 predicted items for first sample: {top_items[0].numpy()}")
            print("✓ Predictions successful")

    except Exception as e:
        print(f"✗ GRU4Rec test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\nGRU4Rec test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_gru_cf()
    sys.exit(0 if success else 1)
