from corerec import cf_engine as cf
from cr_learn import ml_1m as ml


data = ml.load()
cfg = 'examples/ContentFilterExamples/context_config.json'  

users_df = data['users']
ratings_df = data['ratings']
movies_df = data['movies']
user_interactions = data['user_interactions']
item_features = data['item_features']

print("Initializing recommenders...")
user_recommender = cf.CON_USER_PROFILING(user_attributes=users_df)
context_recommender = cf.CON_CONTEXT_AWARE(
    context_config_path=cfg,
    item_features=item_features
)
item_recommender = cf.CON_ITEM_PROFILING()

print("Fitting User Profiling Recommender...")
user_recommender.fit(user_interactions)

print("Fitting Context Aware Recommender...")
context_recommender.fit(user_interactions)

print("Fitting Item Profiling Recommender...")
item_recommender.fit(user_interactions, item_features)

# inf
user_id = 5  
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
recommended_movies = movies_df[movies_df['movie_id'].isin(recommendations)]
print(f"Top 10 recommendations for User {user_id} in context {current_context}:")
for _, row in recommended_movies.iterrows():
    print(f"- {row['title']}")
    