"""
Modern RecSys Pipeline Example

Demonstrates the complete flow:
1. Retrieval with Two-Tower
2. Ranking with DCN
3. Reranking with business rules

This is how production systems work at scale.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from typing import Dict, List

# Import CoreRec components
from corerec.engines import TwoTower
from corerec.retrieval.vector_store import create_index


def generate_synthetic_data(n_users=1000, n_items=10000, density=0.01):
    """Create fake interaction data for demo purposes."""
    
    print(f"Generating synthetic data: {n_users} users, {n_items} items")
    
    # random interactions
    interactions = np.random.rand(n_users, n_items)
    interactions = (interactions < density).astype(np.float32)
    
    # user/item features (one-hot for simplicity)
    user_features = np.eye(n_users, dtype=np.float32)
    item_features = np.eye(n_items, dtype=np.float32)
    
    user_ids = [f"user_{i}" for i in range(n_users)]
    item_ids = [f"item_{i}" for i in range(n_items)]
    
    print(f"Generated {int(interactions.sum())} interactions")
    
    return user_ids, item_ids, interactions, user_features, item_features


def demo_two_tower_retrieval():
    """Demo: fast retrieval with two-tower model."""
    
    print("\n" + "="*60)
    print("DEMO: Two-Tower Retrieval")
    print("="*60)
    
    # small dataset for speed
    user_ids, item_ids, interactions, user_feats, item_feats = generate_synthetic_data(
        n_users=500, n_items=5000, density=0.02
    )
    
    # train two-tower model
    print("\nTraining Two-Tower model...")
    model = TwoTower(
        user_input_dim=user_feats.shape[1],
        item_input_dim=item_feats.shape[1],
        embedding_dim=128,
        hidden_dims=[256, 128],
        loss_type="bce",
        num_epochs=5,
        batch_size=64,
        verbose=True
    )
    
    model.fit(
        user_ids=user_ids,
        item_ids=item_ids,
        interactions=interactions,
        user_features=user_feats,
        item_features=item_feats
    )
    
    # build vector index
    print("\nBuilding vector index...")
    item_embeddings = model.get_item_embeddings()
    
    index = create_index(
        backend="numpy",  # use numpy for demo (no extra deps)
        dim=item_embeddings.shape[1],
        metric="cosine"
    )
    
    index.add(item_embeddings, item_ids)
    
    # retrieve candidates
    print("\nRetrieving candidates for user_0...")
    test_user = user_ids[0]
    user_emb = model.get_user_embedding(test_user)
    
    if user_emb is not None:
        scores, candidates = index.search(user_emb, k=20)
        
        print(f"\nTop 20 candidates for {test_user}:")
        for i, (item_id, score) in enumerate(zip(candidates, scores)):
            print(f"  {i+1}. {item_id}: {score:.4f}")
    
    print("\n✓ Two-Tower retrieval complete")


def demo_sequential_model():
    """Demo: sequential recommendation with BERT4Rec."""
    
    print("\n" + "="*60)
    print("DEMO: Sequential Recommendation")
    print("="*60)
    
    try:
        from corerec.engines import BERT4Rec
        
        # generate sequential data
        user_ids, item_ids, interactions, _, _ = generate_synthetic_data(
            n_users=200, n_items=1000, density=0.05
        )
        
        print("\nTraining BERT4Rec...")
        model = BERT4Rec(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            max_len=50,
            num_epochs=3,
            batch_size=32,
            verbose=True
        )
        
        model.fit(user_ids, item_ids, interactions)
        
        # get recommendations
        test_user = user_ids[0]
        print(f"\nGenerating sequence-aware recommendations for {test_user}...")
        recs = model.recommend(test_user, top_k=10)
        
        print(f"\nTop 10 recommendations:")
        for i, item_id in enumerate(recs):
            print(f"  {i+1}. {item_id}")
        
        print("\n✓ Sequential model demo complete")
    
    except Exception as e:
        print(f"\n⚠ BERT4Rec demo skipped: {e}")


def demo_multimodal_fusion():
    """Demo: combining multiple modalities."""
    
    print("\n" + "="*60)
    print("DEMO: Multi-Modal Fusion")
    print("="*60)
    
    try:
        from corerec.multimodal.fusion_strategies import MultiModalFusion
        
        batch_size = 16
        
        # simulate multi-modal item features
        text_embeddings = torch.randn(batch_size, 768)  # from BERT
        image_embeddings = torch.randn(batch_size, 2048)  # from ResNet
        metadata = torch.randn(batch_size, 32)  # categories, price, etc
        
        print("\nCreating multi-modal fusion layer...")
        fusion = MultiModalFusion(
            modality_dims={
                'text': 768,
                'image': 2048,
                'metadata': 32
            },
            output_dim=256,
            strategy='attention'
        )
        
        print("Fusing modalities...")
        fused = fusion({
            'text': text_embeddings,
            'image': image_embeddings,
            'metadata': metadata
        })
        
        print(f"\nInput shapes:")
        print(f"  Text: {text_embeddings.shape}")
        print(f"  Image: {image_embeddings.shape}")
        print(f"  Metadata: {metadata.shape}")
        print(f"\nOutput shape: {fused.shape}")
        print(f"Successfully fused {len(fusion.modality_dims)} modalities into {fused.shape[1]}D space")
        
        print("\n✓ Multi-modal fusion demo complete")
    
    except Exception as e:
        print(f"\n⚠ Multi-modal demo skipped: {e}")


def demo_complete_pipeline():
    """
    Demo: full pipeline with all three stages.
    This is simplified but shows the architecture.
    """
    
    print("\n" + "="*60)
    print("DEMO: Complete Pipeline Architecture")
    print("="*60)
    
    print("\nIn production, the pipeline looks like:")
    print("\n  User Request")
    print("       ↓")
    print("  [1] RETRIEVAL")
    print("      - Two-Tower model")
    print("      - FAISS index with 10M items")
    print("      - Returns ~1000 candidates")
    print("      - Latency: 5-10ms")
    print("       ↓")
    print("  [2] RANKING")
    print("      - DCN/DLRM with rich features")
    print("      - Score 1000 candidates")
    print("      - Returns top 100")
    print("      - Latency: 20-50ms")
    print("       ↓")
    print("  [3] RERANKING")
    print("      - Business rules")
    print("      - Diversity constraints")
    print("      - Freshness boost")
    print("      - Returns final 10")
    print("      - Latency: 5ms")
    print("       ↓")
    print("  Final Recommendations")
    
    print("\nTotal latency: 30-65ms for 10M item catalog")
    print("This scales to billions of items with proper infrastructure.")
    
    print("\n✓ Pipeline overview complete")


def main():
    """Run all demos."""
    
    print("="*60)
    print("CoreRec Modern RecSys Demo")
    print("="*60)
    print("\nThis demonstrates the evolution from matrix factorization")
    print("to modern embedding-based deep learning systems.")
    
    # set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # run demos
    demo_two_tower_retrieval()
    demo_sequential_model()
    demo_multimodal_fusion()
    demo_complete_pipeline()
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Read MODERN_RECSYS_GUIDE.md for detailed explanation")
    print("2. Try with your own data")
    print("3. Experiment with different fusion strategies")
    print("4. Scale up with FAISS for production")
    
    print("\nFor questions: https://github.com/vishesh9131/CoreRec")


if __name__ == "__main__":
    main()

