import numpy as np
import core_rec as cs
import vish_graphs as vg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Generate random graph and load adjacency matrix
file_path = vg.generate_random_graph(40, seed=122)
adj_matrix = np.loadtxt(file_path, delimiter=",")
top_nodes = vg.find_top_nodes(adj_matrix)
# vg.draw_graph(adj_matrix, top_nodes=top_nodes)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Convert adjacency matrix to a PyTorch tensor of dtype float32
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

# Initialize Transformer Model
num_layers = 3
d_model = 128
num_heads = 4
d_feedforward = 256
input_dim = adj_matrix.shape[1]  # Input dimension should match the number of features per node
hidden_dim = 64  # Define hidden layer dimension
output_dim = adj_matrix.shape[1]  # Output dimension should match the input dimension

# model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
model = SimpleNN(input_dim, hidden_dim, output_dim)

dataset = cs.GraphDataset(adj_matrix)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define your loss function, optimizer, and other training parameters
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 150

# Train the model
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

# Use the trained model for node recommendations
node_index = 2
predictions = cs.predict(model, adj_matrix, node_index, top_k=5)
print(f"Recommended nodes for node {node_index}: {predictions}")
print("Popular Nodes are:", top_nodes)

vg.draw_graph_3d(adj_matrix, top_nodes=top_nodes, recommended_nodes=predictions,transparent_labeled=False)