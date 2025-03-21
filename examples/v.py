# Import necessary modules
import torch
from torch.utils.data import DataLoader
import pandas as pd

from cr_learn.ml_1m import load
from corerec.cf_engine import MIND 

data = load()

users_df = data['users']
ratings_df = data['ratings']
movies_df = data['movies']
user_interactions = data['user_interactions']
item_features = data['item_features']


# Define vocab_size and embedding_dim before initializing tra_mind
vocab_size = len(movies_df)  # or another appropriate value
embedding_dim = 64  # or another appropriate value

# Initialize tra_mind with the required arguments
tra_mind_model = MIND(vocab_size=vocab_size, embedding_dim=embedding_dim)

def prepare_data(user_interactions, vocab_size):
    user_ids = list(user_interactions.keys())
    data = []
    for user_id in user_ids:
        movie_ids = user_interactions[user_id]
        movie_ids = [min(movie_id, vocab_size - 1) for movie_id in movie_ids]
        movie_ids_tensor = torch.zeros(vocab_size, dtype=torch.long)
        movie_ids_tensor[:len(movie_ids)] = torch.tensor(movie_ids, dtype=torch.long)
        data.append((movie_ids_tensor, torch.tensor(user_id, dtype=torch.long)))
    return data

vocab_size = len(movies_df)
data = prepare_data(user_interactions, vocab_size)
dataloader = DataLoader(data, batch_size=32, shuffle=True)

def train(model, dataloader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification

    for epoch in range(epochs):
        for batch in dataloader:
            movie_ids, user_ids = batch
            optimizer.zero_grad()
            outputs = model(movie_ids)
            labels = movie_ids[:, 0]  # Use the first movie_id as the target
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train(tra_mind_model, dataloader)

def recommend(model, user_interactions, top_k=5):
    with torch.no_grad():
        user_tensor = torch.tensor(user_interactions, dtype=torch.long).unsqueeze(0)
        scores = model(user_tensor)
        _, recommended_indices = torch.topk(scores, top_k, dim=1)
        return recommended_indices.squeeze().tolist()

user_id = 5

if user_id is not None and user_id in user_interactions:
    user_interactions = user_interactions[user_id]
    recommendations = recommend(tra_mind_model, user_interactions)
    print(f'Recommendations for user {user_id}: {recommendations}')
    
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommendations)]
    if not recommended_movies.empty:
        print(f"Top {len(recommendations)} recommendations for User {user_id}:")
        for _, row in recommended_movies.iterrows():
            print(f"- {row['title']}")
    else:
        print("No recommended movies found in the dataset.")
else:
    print("No valid user ID found in the dataset.")


