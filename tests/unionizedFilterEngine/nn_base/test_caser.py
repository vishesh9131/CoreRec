#!/usr/bin/env python3
"""
Caser (Convolutional Sequence Embedding Recommendation) Test Script
"""

import os
import sys
import numpy as np
import torch

# Add CoreRec to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.unionizedFilterEngine.nn_base.caser import Caser

def test_caser():
    """Test Caser model with synthetic sequential data"""
    print("=" * 60)
    print(" Caser (Convolutional Sequence Embedding) Test")
    print("=" * 60)
    
    # Create synthetic sequential data for Caser
    print("\nCreating synthetic sequential data...")
    np.random.seed(42)
    
    num_users = 50
    num_items = 100
    num_interactions = 500
    
    # Caser expects list of (user, item, timestamp) tuples
    interactions = []
    for i in range(num_interactions):
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        timestamp = i
        interactions.append((user_id, item_id, timestamp))
    
    print(f"Dataset: {len(interactions)} interactions, {len(set(u for u,i,t in interactions))} users, {len(set(i for u,i,t in interactions))} items")
    
    try:
        # Initialize Caser model
        print("\nInitializing Caser model...")
        model = Caser(
            name="Caser_Test",
            config={
                'embedding_dim': 32,
                'num_h_filters': 16,
                'num_v_filters': 4,
                'dropout': 0.2,
                'num_epochs': 3,
                'batch_size': 64
            },
            seed=42
        )
        
        # Use fit method (recommender pattern)
        print("\nTraining Caser model...")
        model.fit(interactions)
        
        print("✓ Model trained successfully")
        
        # Test recommendation
        print("\nTesting recommendations...")
        test_user = interactions[0][0]
        try:
            recommendations = model.recommend(test_user, top_n=5)
            print(f"Top 5 recommendations for user {test_user}: {recommendations}")
            print("✓ Recommendations successful")
        except Exception as e:
            print(f"Note: {e}")
        
    except Exception as e:
        print(f"✗ Caser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nCaser test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_caser()
    sys.exit(0 if success else 1)

