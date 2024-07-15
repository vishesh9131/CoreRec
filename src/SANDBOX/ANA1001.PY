# Code_ID : ANA1001.PY 
# analysis_log_ANA : https://docs.google.com/document/d/1nnihzGFsfJPsShpToPXRthSvCw1dQmRV4lEwGycxP9U/edit?usp=sharing

import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import numpy as np
import vish_graphs as vg
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import core_rec as cs
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
adj_matrix = np.loadtxt('SANDBOX/adj.csv', delimiter=",")
wgt_matrix = np.loadtxt('SANDBOX/label.csv', delimiter=",")

# Load node labels
df = pd.read_csv("SANDBOX/labelele.csv")
col = df.values.flatten()
node_labels = {i: label for i, label in enumerate(col)}

# Find top nodes
top_nodes = vg.find_top_nodes(adj_matrix, 4)

# ML
# Convert adjacency matrix to dataset
graph_dataset = cs.GraphDataset(adj_matrix)
data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

# Define model parameters
num_layers = 2
d_model = 128 #embedding dimension
num_heads_list = [2, 4, 8]  # Different values of num_heads to test
d_feedforward = 512
input_dim = len(adj_matrix[0])
num_epochs = 100
node_index = 7  # target node

# Initialize storage for recommendations
recommendations_count = {num_heads: {'6': 0, '8': 0} for num_heads in num_heads_list}

for num_heads in num_heads_list:
    delta = 0
    Warning = False

    if d_model % num_heads != 0:
        Warning = True
        delta = d_model % num_heads
        d_model = d_model - delta

    # Initialize model, loss function, and optimizer
    model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim, use_weights=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=5)

    # Train the model and record recommendations
    for epoch in range(num_epochs):
        for batch in data_loader:
            batch = [torch.tensor(item).float() for item in batch]
            batch = torch.stack(batch)  # Stack the list of tensors into a single tensor
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

        # Predict recommendations for the specific node
        recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=5, threshold=0.5)
        
        # Count occurrences of nodes 6 and 8
        if 6 in recommended_nodes:
            recommendations_count[num_heads]['6'] += 1
        if 8 in recommended_nodes:
            recommendations_count[num_heads]['8'] += 1

# Plot the recommendations count
labels = ['6', '8']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
for i, num_heads in enumerate(num_heads_list):
    counts = [recommendations_count[num_heads]['6'], recommendations_count[num_heads]['8']]
    ax.bar(x + i * width, counts, width, label=f'num_heads={num_heads}')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Nodes')
ax.set_ylabel('Count')
ax.set_title('Count of Recommendations for Nodes 6 and 8 to Node 7')
ax.set_xticks(x + width * (len(num_heads_list) - 1) / 2)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()