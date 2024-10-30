import torch
from Tmodel import GraphTransformerV2
from preprocess_actors import load_and_preprocess, ActorDataset
from torch.utils.data import DataLoader

# Load and preprocess the dataset
adjacency_matrix, features, actors_df = load_and_preprocess('src/SANDBOX/dataset/BollywoodActorRanking.csv')
num_actors = features.size(0)
input_dim = features.size(1)  # 6 features

# Initialize the model
d_model = 16
num_layers = 3
num_heads = 4
d_feedforward = 64
model = GraphTransformerV2(
    num_layers=num_layers, 
    d_model=d_model, 
    num_heads=num_heads, 
    d_feedforward=d_feedforward, 
    input_dim=input_dim
)

# Create dataset and dataloader
dataset = ActorDataset(adjacency_matrix, features)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Model initialized with input dimension:", input_dim)
print("Number of actors:", num_actors)
print("Data Loader Ready.")