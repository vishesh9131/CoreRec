import pandas as pd
from numpy import dot
from numpy.linalg import norm

# Old Import technique
# from corerec.engines.contentFilterEngine.context_personalization import (
#     cr.CON_CONTEXT_AWARE,
#     cr.CON_USER_PROFILING,
#     cr.CON_ITEM_PROFILING
# )
# from corerec.engines.contentFilterEngine.embedding_representation_learning import (
#     cr.cr.EMB_PERSONALIZED_EMBEDDINGS
# )
from typing import Dict, List, Any
import os
import json
from cr_learn import ml_1m as ml

# New Import technique
from corerec import cf_engine as cr

data = ml.load()
cfg = 'examples/ContentFilterExamples/context_config.json'  
embedding_data = ml.prepare_embedding_data()


# Initialize Recommenders
users_df = data['users']
ratings_df = data['ratings']
movies_df = data['movies']
user_interactions = data['user_interactions']
item_features = data['item_features']

all_items = set(movies_df['movie_id'].tolist())

print("Initializing recommenders...")
user_recommender = cr.CON_USER_PROFILING(user_attributes=users_df)
context_recommender = cr.CON_CONTEXT_AWARE(
    context_config_path=cfg,
    item_features=item_features
)
item_recommender = cr.CON_ITEM_PROFILING()

# Initialize Embedding Models
print("Initializing embedding models...")
embedding_recommender = cr.EMB_PERSONALIZED_EMBEDDINGS()

# Prepare data for embeddings
print("Preparing data for embeddings...")
embedding_sentences = embedding_data

# Train Embedding Models
print("Training Word2Vec model...")
embedding_recommender.train_word2vec(sentences=embedding_sentences, epochs=10)

print("Training Doc2Vec model...")
embedding_recommender.train_doc2vec(documents=embedding_sentences)

# Fit Recommenders
print("Fitting User Profiling Recommender...")
user_recommender.fit(user_interactions)

print("Fitting Context Aware Recommender...")
context_recommender.fit(user_interactions)

print("Fitting Item Profiling Recommender...")
item_recommender.fit(user_interactions, item_features)

# Generate Embedding-Based Recommendations (Example)
user_id = 1  # Replace with desired user ID
current_context = {
    "time_of_day": "evening",
    "location": "home"
}

print(f"Generating recommendations for User {user_id} with context {current_context} using embeddings...")
# Example: Get user profile attributes
user_profile = user_recommender.user_profiles.get(user_id, {})
if not user_profile:
    print(f"No profile found for User {user_id}.")
    recommendations = []
else:
    # Example: Aggregate embeddings based on user interacted items
    user_embedding = {}
    interacted_items = user_profile.get('interacted_items', set())
    for item_id in interacted_items:
        genres = movies_df[movies_df['movie_id'] == item_id]['genres'].values[0].split('|')
        for genre in genres:
            genre_embedding = embedding_recommender.get_word_embedding(genre)
            for idx, val in enumerate(genre_embedding):
                user_embedding[idx] = user_embedding.get(idx, 0.0) + val
    # Compute average embedding
    num_genres = len(interacted_items) * len(genres) if interacted_items else 1
    user_embedding = {k: v / num_genres for k, v in user_embedding.items()}

    scores = {}
    for item_id in all_items:
        if item_id in interacted_items:
            continue
        genres = movies_df[movies_df['movie_id'] == item_id]['genres'].values[0].split('|')
        item_embedding = []
        for genre in genres:
            genre_emb = embedding_recommender.get_word_embedding(genre)
            item_embedding.extend(genre_emb)
        # Simple scoring: dot product of user and item embeddings
        if not item_embedding:
            continue
        item_vector = sum(item_embedding)
        user_vector = sum(user_embedding.values())
        similarity = dot([user_vector], [item_vector]) / (norm([user_vector]) * norm([item_vector]) + 1e-10)
        scores[item_id] = similarity

    # Sort and get top-N
    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [item_id for item_id, score in ranked_items[:10]]

# Fetch and display movie titles for recommended movie IDs
if recommendations:
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommendations)]
    print(f"Top 10 embedding-based recommendations for User {user_id} in context {current_context}:")
    for _, row in recommended_movies.iterrows():
        print(f"- {row['title']}")
else:
    print("No recommendations could be generated.")

