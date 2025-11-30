#!/usr/bin/env python3
"""
NextItNet Test Script
"""

from corerec.engines.unionizedFilterEngine.nn_base.nextitnet import NextItNetModel
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


def test_nextitnet():
    """Test NextItNet model with synthetic sequential data"""
    print("=" * 60)
    print(" NextItNet (Next Item Recommendation) Test")
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
        # Initialize NextItNet model
        print("\nInitializing NextItNet model...")
        model = NextItNetModel(
            n_items=num_items, embedding_dim=32, dilations=[
                1, 2, 4], kernel_size=3, dropout=0.2)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Create synthetic batch
        sequences = torch.randint(0, num_items, (batch_size, seq_length))
        targets = torch.randint(0, num_items, (batch_size,))

        # Simple training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(sequences)  # (batch_size, seq_len, n_items)
            # Take last timestep for prediction
            outputs_last = outputs[:, -1, :]  # (batch_size, n_items)
            loss = criterion(outputs_last, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/3, Loss: {loss.item():.4f}")

        print("✓ Model trained successfully")

        # Test prediction
        print("\nTesting predictions...")
        model.eval()
        with torch.no_grad():
            test_seqs = sequences[:5]
            predictions_3d = model(test_seqs)  # (5, seq_len, n_items)
            predictions = torch.softmax(
                predictions_3d[:, -1, :], dim=1)  # Take last timestep
            top_items = torch.topk(predictions, k=5, dim=1)[1]
            print(
                f"Top-5 predicted items for first sample: {top_items[0].numpy()}")
            print("✓ Predictions successful")

    except Exception as e:
        print(f"✗ NextItNet test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\nNextItNet test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_nextitnet()
    sys.exit(0 if success else 1)
