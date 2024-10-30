# test3 here tries to recommend 
# similar actors but failed due to 
# disabled dataset

import torch
import pandas as pd
from Tmodel import GraphTransformerV2
import torch.nn.functional as F
# Load the dataset
bolly_df = pd.read_csv('src/SANDBOX/dataset/BollywoodActorRanking.csv')

# Preprocess the dataset to create a graph
num_books = len(bolly_df)
adjacency_matrix = torch.eye(num_books)  # Identity matrix as a placeholder
graph_metrics = torch.rand(num_books, num_books)  # Random metrics as a placeholder

# Initialize the model
input_dim = 301  # Example input dimension
d_model = 8
num_layers = 2
num_heads = 2
d_feedforward = 16
model = GraphTransformerV2(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_feedforward=d_feedforward, input_dim=input_dim)

# Ensure x has the correct number of rows (num_books) and columns (input_dim)
x = torch.rand(num_books, input_dim)

# Train the model (simplified)
output = model(x, adjacency_matrix, graph_metrics)

# Initialize user feedback storage
user_feedback = torch.zeros(num_books)

def recommend_actors():
    print("Welcome to the Bollywood Actor Recommendation System!")
    liked_actors = set()  # Track liked actors

    while True:
        if not liked_actors:
            user_input = input("Enter an actor name you like (or 'bye' to quit): ").strip()
            if user_input.lower() == 'bye':
                break

            # Find the actor in the dataset
            actor_idx = bolly_df[bolly_df['actorName'].str.contains(user_input, case=False, na=False)].index
            if len(actor_idx) == 0:
                print("Actor not found. Please try another name.")
                continue

            # Get recommendations based on the model's output
            scores = output[actor_idx].detach()
            recommended_idx = scores.argsort(dim=1, descending=True)[0][:5]  # Get top 5 recommendations
        else:
            # Recommend more actors similar to liked ones using cosine similarity
            print("Based on your likes, you might also like:")
            recommended_idx = torch.tensor([], dtype=torch.long)
            for liked_idx in liked_actors:
                liked_actor_output = output[liked_idx].unsqueeze(0)
                similarities = F.cosine_similarity(output, liked_actor_output, dim=1)
                similar_recommended_idx = similarities.argsort(descending=True)[:5]  # Get top 5 similar recommendations
                recommended_idx = torch.cat((recommended_idx, similar_recommended_idx))

        print("Recommended actors:")
        for idx in recommended_idx.unique():
            if idx not in liked_actors:  # Avoid recommending already liked actors
                actor_name = bolly_df.iloc[idx.item()]['actorName']  # Convert tensor to integer
                print(f"- {actor_name}")
                feedback = input(f"Do you like this recommendation? (1 for Yes, 0 for No): ").strip()
                if feedback == '1':
                    user_feedback[idx] += 1  # Increment like count
                    liked_actors.add(idx)  # Add to liked actors
                elif feedback == '0':
                    user_feedback[idx] -= 1  # Decrement like count

# Example of using feedback to adjust features
def update_features_with_feedback(x, user_feedback):
    # Normalize feedback and add to features
    feedback_normalized = user_feedback / torch.max(torch.abs(user_feedback))
    x += feedback_normalized.unsqueeze(1)  # Add feedback as a new feature dimension
    return x

# Update features with feedback
x = update_features_with_feedback(x, user_feedback)

# Optionally, retrain the model with updated features
# output = model(x, adjacency_matrix, graph_metrics)

# Start the recommendation system
recommend_actors()
