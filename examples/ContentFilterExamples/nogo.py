from corerec.engines.content_based.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_ITEM_PROFILING,
    CON_USER_PROFILING
)
from cr_learn import ml_1m as ml

# dataset call
data = ml.load()  
cfg = 'examples/ContentFilterExamples/context_config.json'

users_df=data['users'] 
ratings_df=data['ratings']
movies_df=data['movies']
user_interactions=data['user_interactions']
item_features=data['item_features']

# recommenders initilize..
usr_rec= CON_USER_PROFILING(
    user_attributes=users_df
)
con_rec=CON_CONTEXT_AWARE(
    context_config_path=cfg,
    item_features=item_features
)
item_rec= CON_ITEM_PROFILING()

# fit
usr_rec.fit(user_interactions)
con_rec.fit(user_interactions)
item_rec.fit(user_interactions,item_features)

# testing 
user_id=5
current_context={
    "time_of_day":"evening",
    "location":"home"
}
rec=con_rec.recommend(user_id=1,
                      context=current_context,
                      top_n=10
                      )
rec_movies=movies_df[movies_df['movie_id'].isin(rec)]
for _,row in rec_movies.iterrows():
    print(f"- {row['title']}")