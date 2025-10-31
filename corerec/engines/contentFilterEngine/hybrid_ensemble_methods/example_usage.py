"""
Example Usage of Hybrid & Ensemble Methods

This script demonstrates how to use the hybrid and ensemble recommendation methods
provided in this module.

Author: Vishesh Yadav
"""

import numpy as np
from attention_mechanisms import ATTENTION_MECHANISMS
from ensemble_methods import ENSEMBLE_METHODS
from hybrid_collaborative import HYBRID_COLLABORATIVE


def create_sample_data(n_users=100, n_items=50, n_features=20, density=0.1):
    """Create sample data for testing"""
    # create user-item interaction matrix (sparse)
    user_item_matrix = np.zeros((n_users, n_items))
    n_interactions = int(n_users * n_items * density)
    
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)
        rating = np.random.randint(1, 6)  # ratings from 1-5
        user_item_matrix[user_id, item_id] = rating
    
    # create item content features
    item_features = np.random.randn(n_items, n_features)
    
    return user_item_matrix, item_features


def example_attention_mechanisms():
    """Example of using attention mechanisms"""
    print("=" * 70)
    print("ATTENTION MECHANISMS EXAMPLE")
    print("=" * 70)
    
    # create sample data
    user_item_matrix, _ = create_sample_data(n_users=50, n_items=30)
    
    # initialize attention model
    attention_model = ATTENTION_MECHANISMS(
        num_heads=4,
        embedding_dim=64,
        dropout_rate=0.1,
        random_state=42
    )
    
    # train the model
    print("\nTraining attention-based model...")
    attention_model.train(
        user_item_matrix=user_item_matrix,
        epochs=5,
        learning_rate=0.01,
        verbose=True
    )
    
    # generate recommendations
    user_id = 0
    known_items = np.where(user_item_matrix[user_id, :] > 0)[0]
    
    print(f"\nGenerating recommendations for user {user_id}...")
    recommended_items, scores = attention_model.recommend(
        user_id=user_id,
        top_n=5,
        exclude_known=True,
        known_items=known_items
    )
    
    print("\nTop-5 Recommendations:")
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        print(f"  {i+1}. Item {item_id}: Score = {score:.4f}")
    
    # get attention weights for specific user-item pair
    if len(recommended_items) > 0:
        item_id = recommended_items[0]
        attention_weights = attention_model.get_attention_weights(user_id, item_id)
        print(f"\nAttention weights for user {user_id} and item {item_id}:")
        print(f"  Shape: {attention_weights.shape}")
        print(f"  Mean weight: {np.mean(attention_weights):.4f}")


def example_ensemble_methods():
    """Example of using ensemble methods"""
    print("\n\n" + "=" * 70)
    print("ENSEMBLE METHODS EXAMPLE")
    print("=" * 70)
    
    # create sample data
    user_item_matrix, item_features = create_sample_data(n_users=50, n_items=30)
    
    # create some simple mock models for the ensemble
    class SimpleModel:
        """Simple mock model for demonstration"""
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level
            
        def recommend(self, user_id, top_n=10):
            # generate random recommendations with some noise
            n_items = 30
            scores = np.random.rand(n_items) + np.random.randn(n_items) * self.noise_level
            top_indices = np.argsort(scores)[::-1][:top_n]
            return top_indices, scores[top_indices]
        
        def predict(self, user_id, item_id):
            return np.random.rand() * 5  # random rating 0-5
    
    # create ensemble
    ensemble = ENSEMBLE_METHODS(
        ensemble_strategy='weighted_average',
        normalize_scores=True,
        random_state=42
    )
    
    # add models to ensemble
    print("\nAdding models to ensemble...")
    ensemble.add_model(SimpleModel(noise_level=0.05), name="Model_A", weight=1.5)
    ensemble.add_model(SimpleModel(noise_level=0.10), name="Model_B", weight=1.0)
    ensemble.add_model(SimpleModel(noise_level=0.15), name="Model_C", weight=0.5)
    
    # train ensemble (learns weights)
    print("Training ensemble...")
    ensemble.train(verbose=True)
    
    # generate ensemble recommendations
    user_id = 0
    print(f"\nGenerating ensemble recommendations for user {user_id}...")
    recommended_items, scores = ensemble.recommend(
        user_id=user_id,
        top_n=5,
        return_scores=True
    )
    
    print("\nTop-5 Ensemble Recommendations:")
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        print(f"  {i+1}. Item {item_id}: Score = {score:.4f}")
    
    # get individual model contributions
    test_item_id = 5
    contributions = ensemble.get_model_contributions(user_id, test_item_id)
    print(f"\nModel contributions for user {user_id}, item {test_item_id}:")
    for model_name, pred in contributions.items():
        if pred is not None:
            print(f"  {model_name}: {pred:.4f}")
    
    # evaluate diversity
    diversity = ensemble.evaluate_diversity()
    print(f"\nEnsemble diversity metrics:")
    print(f"  Number of models: {diversity['num_models']}")
    print(f"  Model names: {diversity['model_names']}")
    print(f"  Model weights: {diversity['weights']}")


def example_hybrid_collaborative():
    """Example of using hybrid collaborative filtering"""
    print("\n\n" + "=" * 70)
    print("HYBRID COLLABORATIVE FILTERING EXAMPLE")
    print("=" * 70)
    
    # create sample data
    user_item_matrix, item_features = create_sample_data(n_users=50, n_items=30)
    
    # initialize hybrid model
    hybrid_model = HYBRID_COLLABORATIVE(
        hybrid_strategy='weighted',
        cf_weight=0.6,
        content_weight=0.4,
        cf_method='item_based',
        similarity_metric='cosine',
        k_neighbors=10,
        random_state=42
    )
    
    # train the model
    print("\nTraining hybrid model...")
    hybrid_model.train(
        user_item_matrix=user_item_matrix,
        content_features=item_features,
        compute_similarities=True,
        verbose=True
    )
    
    # generate recommendations
    user_id = 0
    print(f"\nGenerating hybrid recommendations for user {user_id}...")
    recommended_items, scores = hybrid_model.recommend(
        user_id=user_id,
        top_n=5,
        exclude_known=True,
        return_scores=True
    )
    
    print("\nTop-5 Hybrid Recommendations:")
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        print(f"  {i+1}. Item {item_id}: Score = {score:.4f}")
    
    # evaluate component contributions
    if len(recommended_items) > 0:
        test_item = recommended_items[0]
        components = hybrid_model.evaluate_components(user_id, test_item)
        print(f"\nComponent analysis for user {user_id}, item {test_item}:")
        print(f"  CF prediction: {components['cf_prediction']:.4f}")
        print(f"  Content prediction: {components['content_prediction']:.4f}")
        print(f"  Hybrid prediction: {components['hybrid_prediction']:.4f}")
        print(f"  Global mean: {components['global_mean']:.4f}")
    
    # get similar items
    target_item = 5
    print(f"\nFinding items similar to item {target_item}...")
    
    # using collaborative filtering similarity
    similar_items_cf, similarities_cf = hybrid_model.get_similar_items(
        item_id=target_item,
        top_n=5,
        use_content=False
    )
    print("\nTop-5 similar items (CF-based):")
    for i, (item_id, sim) in enumerate(zip(similar_items_cf, similarities_cf)):
        print(f"  {i+1}. Item {item_id}: Similarity = {sim:.4f}")
    
    # using content similarity
    similar_items_content, similarities_content = hybrid_model.get_similar_items(
        item_id=target_item,
        top_n=5,
        use_content=True
    )
    print("\nTop-5 similar items (Content-based):")
    for i, (item_id, sim) in enumerate(zip(similar_items_content, similarities_content)):
        print(f"  {i+1}. Item {item_id}: Similarity = {sim:.4f}")


def example_comparison():
    """Compare all three methods on the same dataset"""
    print("\n\n" + "=" * 70)
    print("COMPARING ALL METHODS")
    print("=" * 70)
    
    # create shared dataset
    user_item_matrix, item_features = create_sample_data(
        n_users=50, 
        n_items=30,
        density=0.15
    )
    
    user_id = 0
    top_n = 5
    
    print(f"\nComparing recommendations for user {user_id}...")
    print(f"User has rated {np.sum(user_item_matrix[user_id, :] > 0)} items")
    
    # 1. Attention-based
    print("\n1. Attention Mechanisms:")
    attention_model = ATTENTION_MECHANISMS(num_heads=2, embedding_dim=32, random_state=42)
    attention_model.train(user_item_matrix, epochs=3, verbose=False)
    attn_items, attn_scores = attention_model.recommend(
        user_id, top_n=top_n,
        known_items=np.where(user_item_matrix[user_id, :] > 0)[0]
    )
    for i, (item, score) in enumerate(zip(attn_items, attn_scores)):
        print(f"   {i+1}. Item {item}: {score:.4f}")
    
    # 2. Hybrid Collaborative
    print("\n2. Hybrid Collaborative:")
    hybrid_model = HYBRID_COLLABORATIVE(
        hybrid_strategy='weighted',
        cf_weight=0.6,
        content_weight=0.4,
        random_state=42
    )
    hybrid_model.train(user_item_matrix, item_features, verbose=False)
    hybrid_items, hybrid_scores = hybrid_model.recommend(user_id, top_n=top_n)
    for i, (item, score) in enumerate(zip(hybrid_items, hybrid_scores)):
        print(f"   {i+1}. Item {item}: {score:.4f}")
    
    print("\n" + "=" * 70)
    print("Comparison complete! Each method has its own strengths:")
    print("  - Attention: Good for capturing complex patterns")
    print("  - Ensemble: Combines multiple models for robustness")
    print("  - Hybrid: Balances collaborative and content signals")
    print("=" * 70)


if __name__ == "__main__":
    # run all examples
    np.random.seed(42)
    
    print("\n" + "#" * 70)
    print("# HYBRID & ENSEMBLE METHODS - COMPREHENSIVE EXAMPLES")
    print("#" * 70)
    
    # run individual examples
    example_attention_mechanisms()
    example_ensemble_methods()
    example_hybrid_collaborative()
    
    # run comparison
    example_comparison()
    
    print("\n\n" + "#" * 70)
    print("# All examples completed successfully!")
    print("#" * 70 + "\n")

