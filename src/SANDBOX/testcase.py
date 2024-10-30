import pandas as pd
import torch
import numpy as np
import core_rec as cr
import vish_graphs as vg
from corerec.torch_nn import *
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

def train_model123(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")



# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Create a DataLoader for the Cora dataset
data_loader = DataLoader([data], batch_size=1, shuffle=True)

# Define model parameters
num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = dataset.num_node_features

# Initialize model, loss function, and optimizer
model = cr.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
train_model123(model, data_loader, criterion, optimizer, num_epochs)

# Predict recommendations for a specific node
node_index = 2   #target node
recommended_nodes = cr.predict(model, data.x.numpy(), node_index, top_k=5, threshold=0.5)
print(f"Recommended nodes for node {node_index}: {recommended_nodes}")

# Visualize the graph (optional)
vg.draw_graph(data.x.n, top_nodes, recommended_nodes, transparent_labeled=False)