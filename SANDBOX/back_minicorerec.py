import numpy as np
import vish_graphs as vg
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import core_rec as cs
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Load the CSV file into a DataFrame
adj_matrix = np.loadtxt('SANDBOX/label.csv', delimiter=",")

# Load node labels
df = pd.read_csv("SANDBOX/labelele.csv")
col = df.values.flatten()
node_labels = {i: label for i, label in enumerate(col)}
label_to_index = {label: i for i, label in enumerate(col)}  # Create a reverse mapping

# Print available labels for debugging
print("Available labels:", list(label_to_index.keys()))

# Find top nodes
top_nodes = vg.find_top_nodes(adj_matrix, 4)

# ML
# Convert adjacency matrix to dataset
graph_dataset = cs.GraphDataset(adj_matrix)
batch_size = 3
var = 1.0

# Initialize model parameters
num_layers = 1
d_model = 128  # embedding dimension
num_heads = 2
d_feedforward = 512
input_dim = adj_matrix.shape[1]  # Ensure input_dim matches the number of features in adj_matrix
num_weights = 10

# Initialize model, loss function, and optimizer
model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim, use_weights=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create DataLoader
data_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)

# Train the model
num_epochs = 100
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

# Recommend nodes for a target node
target_node_label = "vishesh"  # Example target node label

# Check if the target node label exists
if target_node_label in label_to_index:
    target_node_index = label_to_index[target_node_label]  # Get the index of the target node
    recommended_nodes = cs.predict(model, adj_matrix, target_node_index, top_k=3, threshold=1)

    # Map recommended node indices to labels
    recommended_labels = [node_labels[idx] for idx in recommended_nodes]

    print(f"Recommended nodes for target node {target_node_label}: {recommended_labels}")
else:
    print(f"Error: The label '{target_node_label}' does not exist. Available labels are: {list(label_to_index.keys())}")