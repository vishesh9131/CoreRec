import numpy as np
from collaborative_filtering.als import *
import vish_graphs as vg


# Sample ratings matrix (users x items)
file_path = vg.generate_random_graph(10,file_path='ratings.csv')
ratings = np.loadtxt(file_path, delimiter=',')

# Initialize ALS model
als = ALS(num_factors=3, regularization=0.1, iterations=10)

# Fit the model to the ratings data
als.fit(ratings)

# Make a prediction for a specific user-item pair
user_id = 0
item_id = 2
prediction = als.predict(user_id, item_id)
print(f"Predicted rating for user {user_id} and item {item_id}: {prediction}")

# Get top-N recommendations for a specific user
recommendations = als.recommend(user_id, num_items=3)
print(f"Top 3 recommendations for user {user_id}: {recommendations}")