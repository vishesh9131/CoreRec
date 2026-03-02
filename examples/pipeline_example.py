"""
Multi-stage Recommendation Pipeline Example

Demonstrates how to build a production-style recommendation system
using corerec's pipeline components.

This example shows:
1. Setting up multiple retrievers (collaborative + semantic)
2. Configuring a ranker
3. Adding rerankers for diversity and business rules
4. Running the full pipeline
"""

import numpy as np
import pandas as pd

# === Simulated Data ===
# In production, this comes from your database

NUM_USERS = 1000
NUM_ITEMS = 5000

# generate fake user-item interactions
np.random.seed(42)
interactions = []
for user_id in range(NUM_USERS):
    # each user interacts with 10-50 items
    n_items = np.random.randint(10, 50)
    items = np.random.choice(NUM_ITEMS, n_items, replace=False)
    for item_id in items:
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': np.random.randint(1, 6),
            'timestamp': np.random.randint(0, 1000),
        })

train_df = pd.DataFrame(interactions)

# fake item features
item_texts = [f"Item {i} description with features" for i in range(NUM_ITEMS)]
item_categories = {i: f"category_{i % 10}" for i in range(NUM_ITEMS)}

print(f"Training data: {len(train_df)} interactions")
print(f"Users: {train_df['user_id'].nunique()}, Items: {train_df['item_id'].nunique()}")


# === Setup Retrievers ===

# 1. collaborative filtering retriever using SAR
from corerec.engines.collaborative import SAR
from corerec.retrieval import CollaborativeRetriever

sar_model = SAR(
    col_user='user_id',
    col_item='item_id',
    col_rating='rating',
    col_timestamp='timestamp',
    similarity_type='jaccard',
)
sar_model.fit(train_df)

collab_retriever = CollaborativeRetriever(model=sar_model, name="collab")

# 2. popularity-based retriever for coverage
from corerec.retrieval import PopularityRetriever

item_counts = train_df.groupby('item_id').size()
pop_retriever = PopularityRetriever(name="popularity")
pop_retriever.fit(
    item_ids=list(item_counts.index),
    interaction_counts=list(item_counts.values),
)


# === Setup Ranker ===

from corerec.ranking import PointwiseRanker

# simple ranker that uses retrieval score + popularity
def feature_extractor(item_id, context):
    return {
        'popularity': item_counts.get(item_id, 0) / item_counts.max(),
        'category_match': 1.0 if item_categories.get(item_id) == context.get('pref_category') else 0.0,
    }

def score_fn(feats):
    # blend retrieval score with popularity and category match
    return (
        0.6 * feats.get('retrieval_score', 0) +
        0.2 * feats.get('popularity', 0) +
        0.2 * feats.get('category_match', 0)
    )

ranker = PointwiseRanker(
    score_fn=score_fn,
    feature_extractor=feature_extractor,
    name="blended_ranker",
)
ranker.fit()


# === Setup Rerankers ===

from corerec.reranking import DiversityReranker, BusinessRulesReranker

# diversity by category
diversity_reranker = DiversityReranker(
    lambda_=0.7,
    category_key='category',
)

# business rules
business_reranker = BusinessRulesReranker(name="business")
business_reranker.add_boost(item_id=42, multiplier=2.0)  # promoted item
business_reranker.add_blocklist([999, 998])  # blocked items


# === Build Pipeline ===

from corerec.pipelines import RecommendationPipeline, PipelineConfig

pipeline = RecommendationPipeline(
    config=PipelineConfig(
        retrieval_k=200,
        ranking_k=50,
        final_k=10,
        fusion_strategy='rrf',
    ),
    name="example_pipeline",
)

# add components
pipeline.add_retriever(collab_retriever, weight=1.0)
pipeline.add_retriever(pop_retriever, weight=0.3)
pipeline.set_ranker(ranker)
pipeline.add_reranker(diversity_reranker)
pipeline.add_reranker(business_reranker)

print(f"\nPipeline: {pipeline}")


# === Generate Recommendations ===

# pick a test user
test_user = 5

# user context
context = {
    'user_id': test_user,
    'pref_category': 'category_3',  # user prefers this category
}

print(f"\nGenerating recommendations for user {test_user}...")

result = pipeline.recommend(
    query=test_user,
    context=context,
    top_k=10,
)

print(f"\nResults:")
print(f"  Retrieved: {result.retrieval_candidates} candidates")
print(f"  Ranked: {result.ranking_candidates} candidates")
print(f"  Final: {result.final_candidates} items")
print(f"\nTiming:")
print(f"  Retrieval: {result.retrieval_ms:.2f}ms")
print(f"  Ranking: {result.ranking_ms:.2f}ms")
print(f"  Reranking: {result.reranking_ms:.2f}ms")
print(f"  Total: {result.total_ms:.2f}ms")

print(f"\nTop {len(result)} recommendations:")
for i, (item_id, score) in enumerate(result, 1):
    cat = item_categories.get(item_id, "unknown")
    print(f"  {i}. Item {item_id} (score={score:.4f}, category={cat})")


# === Batch Recommendations ===

print("\n\nBatch recommendation for multiple users...")
test_users = [1, 2, 3, 4, 5]
batch_results = pipeline.recommend_batch(
    queries=test_users,
    contexts=[{'user_id': u} for u in test_users],
    top_k=5,
)

for user, res in zip(test_users, batch_results):
    items = [f"{item_id}" for item_id in res.items[:3]]
    print(f"  User {user}: {', '.join(items)}... ({res.total_ms:.1f}ms)")


# === With Explanations ===

from corerec.explanation import FeatureExplainer

explainer = FeatureExplainer(
    item_features={i: {'category': item_categories[i]} for i in range(NUM_ITEMS)},
    user_preferences={test_user: {'category': ['category_3', 'category_5']}},
    templates={
        'category': "Matches your interest in {value}",
        'default': "Recommended for you",
    }
)

print(f"\n\nExplanations for user {test_user}'s recommendations:")
for item_id, score in result.to_list()[:5]:
    explanation = explainer.explain(item_id, {'user_id': test_user})
    print(f"  Item {item_id}: {explanation.text}")


print("\n\nDone!")
