import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Load the dataset
data_df = pd.read_csv('src/SANDBOX/dataset/BollywoodMovieDetail.csv')

# Simulate user-item interactions
def simulate_user_item_interactions(data_df, num_users=100):
    user_item_pairs = []
    user_map = {user: idx for idx, user in enumerate(range(num_users))}
    item_map = {item: idx for idx, item in enumerate(data_df['imdbId'].unique())}
    
    for user in user_map.keys():
        num_interactions = np.random.randint(1, 10)
        sampled_items = np.random.choice(list(item_map.keys()), num_interactions, replace=False)
        for item in sampled_items:
            user_item_pairs.append((user_map[user], item_map[item]))
    
    return user_item_pairs, user_map, item_map

# Load data and create adjacency matrix
def load_data():
    user_item_pairs, user_map, item_map = simulate_user_item_interactions(data_df)
    
    num_users = len(user_map)
    num_items = len(item_map)
    
    adjacency_matrix = torch.zeros((num_users + num_items, num_users + num_items))
    for user_idx, item_idx in user_item_pairs:
        adjacency_matrix[user_idx, num_users + item_idx] = 1
        adjacency_matrix[num_users + item_idx, user_idx] = 1
    
    return user_item_pairs, adjacency_matrix, num_users, num_items, user_map, item_map

# Define LightGCN model
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices):
        all_users, all_items = self.compute_embeddings()
        user_embeds = all_users[user_indices]
        item_embeds = all_items[item_indices]
        return (user_embeds * item_embeds).sum(dim=1)

    def compute_embeddings(self):
        user_embeds = self.user_embedding.weight
        item_embeds = self.item_embedding.weight
        all_embeds = torch.cat([user_embeds, item_embeds], dim=0)
        
        for _ in range(self.num_layers):
            all_embeds = torch.sparse.mm(self.adjacency_matrix, all_embeds)
        
        user_embeds, item_embeds = torch.split(all_embeds, [self.num_users, self.num_items])
        return user_embeds, item_embeds

    def set_adjacency_matrix(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

# Prepare data
class MovieDataset(Dataset):
    def __init__(self, interactions):
        self.user_item_pairs = interactions

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        return self.user_item_pairs[idx]

# Train LightGCN
def train_lightgcn():
    user_item_pairs, adjacency_matrix, num_users, num_items, user_map, item_map = load_data()
    dataset = MovieDataset(user_item_pairs)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    model.set_adjacency_matrix(adjacency_matrix)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(10):  # Number of epochs
        for user_indices, item_indices in dataloader:
            optimizer.zero_grad()
            predictions = model(user_indices, item_indices)
            labels = torch.ones_like(predictions)  # Assuming all interactions are positive
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
    
    return model, user_map, item_map

# Get recommendations based on a movie title
def get_recommendations_by_title(model, title, item_map, top_k=5):
    # Find the IMDb ID for the given title
    movie_row = data_df[data_df['title'].str.contains(title, case=False, na=False)]
    if movie_row.empty:
        print(f"No movie found with title containing '{title}'.")
        return []

    imdb_id = movie_row.iloc[0]['imdbId']
    item_idx = item_map.get(imdb_id)

    if item_idx is None:
        print(f"No item index found for IMDb ID '{imdb_id}'.")
        return []

    item_embedding = model.item_embedding.weight[item_idx]
    all_item_embeddings = model.item_embedding.weight
    scores = torch.matmul(all_item_embeddings, item_embedding)
    _, recommended_indices = torch.topk(scores, top_k + 1)  # +1 to exclude the input movie itself

    recommended_items = []
    for idx in recommended_indices:
        if idx.item() == item_idx:
            continue  # Skip the input movie itself
        imdb_id = list(item_map.keys())[list(item_map.values()).index(idx.item())]
        movie_details = data_df[data_df['imdbId'] == imdb_id].iloc[0]
        recommended_items.append({
            'title': movie_details['title'],
            'releaseYear': movie_details['releaseYear'],
            'actors': movie_details['actors'],
            'hitFlop': movie_details['hitFlop']
        })
    
    return recommended_items

# Interactive recommendation loop
def interactive_recommendation():
    model, user_map, item_map = train_lightgcn()

    while True:
        title = input("Enter a movie title you like (or 'exit' to quit): ")
        if title.lower() == 'exit':
            break

        recommendations = get_recommendations_by_title(model, title, item_map)
        if not recommendations:
            continue

        print("Recommended items:")
        for item in recommendations:
            print(f"Title: {item['title']}, Year: {item['releaseYear']}, Actors: {item['actors']}, Hit/Flop: {item['hitFlop']}")

        liked_items = set()
        disliked_items = set()

        for item in recommendations:
            feedback = input(f"Do you like the movie '{item['title']}'? (1 for like, 0 for dislike, 'exit' to quit): ")
            if feedback == 'exit':
                return
            elif feedback == '1':
                liked_items.add(item['title'])
            elif feedback == '0':
                disliked_items.add(item['title'])

        # Update the model or user preferences based on feedback
        # This is a placeholder for updating logic
        # You might want to adjust embeddings or retrain the model with new data

# Start the interactive recommendation system
interactive_recommendation()
