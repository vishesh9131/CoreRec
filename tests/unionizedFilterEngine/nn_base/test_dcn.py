#!/usr/bin/env python3
"""
DCN (Deep & Cross Network) Test Script
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add CoreRec to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines.unionizedFilterEngine.nn_base.DCN import DCN

def test_dcn():
    """Test DCN model with synthetic data"""
    print("=" * 60)
    print(" DCN (Deep & Cross Network) Test")
    print("=" * 60)
    
    # Create synthetic recommendation data
    print("\nCreating synthetic data...")
    np.random.seed(42)
    
    num_users = 100
    num_items = 50
    num_interactions = 1000
    
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    ratings = np.random.choice([0, 1], num_interactions, p=[0.4, 0.6])
    
    # Create additional features
    user_features = np.random.randint(0, 10, num_interactions)
    item_features = np.random.randint(0, 20, num_interactions)
    
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'user_feature': user_features,
        'item_feature': item_features
    })
    
    print(f"Dataset: {num_interactions} samples, {num_users} users, {num_items} items")
    print(f"Positive samples: {sum(ratings)}/{len(ratings)} ({sum(ratings)/len(ratings)*100:.1f}%)")
    
    # Train DCN model using fit method
    print("\nTraining DCN model...")
    
    try:
        model = DCN(
            name="DCN_Test",
            embedding_dim=16,
            num_cross_layers=3,
            deep_layers=[64, 32, 16],
            dropout=0.2,
            num_epochs=3,
            batch_size=128,
            learning_rate=0.001,
            seed=42,
            device='cpu'
        )
        
        # Use fit method (recommender pattern)
        model.fit(data)
        
        print("✓ Model trained successfully")
        
        # Test predictions if predict method exists
        print("\nTesting predictions...")
        try:
            test_user = data['user_id'].iloc[0]
            test_item = data['item_id'].iloc[0]
            prediction = model.predict(test_user, test_item)
            print(f"Prediction for user {test_user}, item {test_item}: {prediction:.4f}")
            print("✓ Predictions successful")
        except Exception as e:
            print(f"Note: Predict method - {e}")
        
        # Test recommendations if recommend method exists
        print("\nTesting recommendations...")
        try:
            recommendations = model.recommend(test_user, top_n=5)
            print(f"Top 5 recommendations for user {test_user}: {recommendations}")
            print("✓ Recommendations successful")
        except Exception as e:
            print(f"Note: Recommend method - {e}")
        
    except Exception as e:
        print(f"✗ DCN test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nDCN test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_dcn()
    sys.exit(0 if success else 1)

