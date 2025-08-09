# [the methods used here is in Maintainence]

# Fairness = Everyone gets treated equally.
# Explainability = You can see why something was recommended.
 
import pandas as pd
from corerec.engines.contentFilterEngine.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_USER_PROFILING,
    CON_ITEM_PROFILING
)
from corerec.engines.contentFilterEngine.fairness_explainability import (
    FAI_EXPLAINABLE,
    FAI_FAIRNESS_AWARE,
    FAI_PRIVACY_PRESERVING
)
from cr_learn import ml_1m as ml

data = ml.load()
cfg = 'examples/ContentFilterExamples/context_config.json'  
embedding_data = ml.prepare_embedding_data()


# Initialize Recommenders
users_df = data['users']
ratings_df = data['ratings']
movies_df = data['movies']
user_interactions = data['user_interactions']
item_features = data['item_features']


# Load Movies Data
print("Loading movies data...")
movies_df = data['movies']

# Build Item Features
print("Building item features...")
item_features = data['item_features']

# Initialize Recommenders
print("Initializing recommenders...")
context_recommender = CON_CONTEXT_AWARE(
    context_config_path='src/SANDBOX/dataset/ml-1m/context_config.json',
    item_features=item_features
)
item_recommender = CON_ITEM_PROFILING()

# Initialize Fairness and Explainability Modules
explainable = FAI_EXPLAINABLE()
fairness_aware = FAI_FAIRNESS_AWARE()

# Example User Interactions (Placeholder)
user_interactions = data['user_interactions']

# Fit Recommenders
print("Fitting Context Aware Recommender...")
context_recommender.fit(user_interactions)

print("Fitting Item Profiling Recommender...")
item_recommender.fit(user_interactions, item_features)

# Generate Recommendations for a User
user_id = 1
current_context = {
    "time_of_day": "evening",
    "location": "home"
}

print(f"Generating recommendations for User {user_id} with context {current_context}...")
recommendations = context_recommender.recommend(
    user_id=user_id,
    context=current_context,
    top_n=10
)

# Ensure Fairness
min_length = min(len(v) for v in user_interactions.values()) if isinstance(user_interactions, dict) else None
user_interactions_df = pd.DataFrame({k: v[:min_length] for k, v in user_interactions.items()}) if min_length else pd.DataFrame(user_interactions)

recommendations = fairness_aware.ensure_fairness({user_id: recommendations}, user_interactions_df)

# Generate Explanations
for item_id in recommendations.get(user_id, []):
    title = movies_df.loc[movies_df['movie_id'] == item_id, 'title'].iloc[0]
    explanation = explainable.generate_explanation(user_id, item_id, current_context)
    print(f"Recommendation: {title}\nExplanation: {explanation}")