
import numpy as np
import core_rec as cs
import vish_graphs as vg
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# Generate random graph
file_path = "SANDBOX/Analysis/data_mother/large_network.csv"
file_path = "SANDBOX/Analysis/data_mother/500
df = pd.read_csv(file_path)
adj_matrix = df.values

    # Convert adjacency matrix to dataset
graph_dataset = cs.GraphDataset(adj_matrix)
data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

    # Define model parameters
num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = len(adj_matrix[0])

    # Initialize model, loss function, and optimizer
model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=5)

    # Train the model
num_epochs = 1
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)


    # Predict recommendations for a specific node
node_index = 12453   #target node
recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=5, threshold=0.5)
print(f"Recommended nodes for node {node_index}: {recommended_nodes}")

