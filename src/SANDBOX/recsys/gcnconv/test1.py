import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from corerec.cr_pkg.gcn_conv import GCNConv  # Import GCNConv
from torch import nn
from torch.nn.functional import cosine_similarity

# Load MovieLens data
data = pd.read_csv(
    'src/SANDBOX/dataset/ml-1m/ratings.dat',
    sep="::",
    header=None,
    engine='python',
    names=['user_id', 'movie_id', 'rating', 'timestamp'],
    encoding='ISO-8859-1'
)

# Load movie metadata
movies = pd.read_csv(
    'src/SANDBOX/dataset/ml-1m/movies.dat',
    sep="::",
    header=None,
    engine='python',
    names=['movie_id', 'title', 'genre'],
    encoding='ISO-8859-1'
)

# Create a dictionary to map movie_id to title
movie_id_to_title = dict(zip(movies['movie_id'], movies['title']))

# Map users and movies to unique integer indices
data['user_id'] = data['user_id'] - 1  # Zero-indexed
data['movie_id'] = data['movie_id'] - 1

# Create edge list (user-movie pairs) and edge attributes (ratings)
edge_index = torch.tensor(data[['user_id', 'movie_id']].values.T, dtype=torch.long)
edge_attr = torch.tensor(data['rating'].values, dtype=torch.float)

# Split into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

class MovieGCNRecommender(nn.Module):
    def __init__(self, num_users, num_movies, hidden_dim, out_dim):
        super(MovieGCNRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.movie_embedding = nn.Embedding(num_movies, hidden_dim)
        
        # Define GCN layers
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        
    def forward(self, edge_index):
        # Initialize embeddings for users and movies
        user_emb = self.user_embedding.weight
        movie_emb = self.movie_embedding.weight
        x = torch.cat([user_emb, movie_emb], dim=0)

        # Apply GCN layers
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        return x

def train(model, edge_index, edge_attr, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass: Compute embeddings for nodes
        out = model(edge_index)
        
        # Compute similarity scores for each edge (dot product between node pairs)
        src, dst = edge_index  # Source and destination nodes for each edge
        edge_scores = cosine_similarity(out[src], out[dst], dim=1)
        
        # Compute loss (compare edge_scores with edge_attr)
        loss = F.mse_loss(edge_scores, edge_attr)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Instantiate model and optimizer
num_users = data['user_id'].nunique()
num_movies = data['movie_id'].nunique()
model = MovieGCNRecommender(num_users, num_movies, hidden_dim=64, out_dim=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
train(model, edge_index, edge_attr, optimizer)

def recommend(model, user_id, top_k=5):
    model.eval()
    with torch.no_grad():
        user_emb = model.user_embedding(torch.tensor(user_id)).unsqueeze(0)
        movie_embs = model.movie_embedding.weight

        # Compute scores and recommend top-K movies
        scores = torch.matmul(user_emb, movie_embs.T).squeeze(0)
        recommended_movies = scores.topk(top_k).indices
        return recommended_movies

# Example recommendation for a specific user
user_id = 0
top_k = 5
recommended_movies = recommend(model, user_id, top_k=top_k)

# Translate recommended IDs to titles
recommended_titles = [movie_id_to_title[movie_id.item()] for movie_id in recommended_movies]
print("Recommended Movie Titles:", recommended_titles)