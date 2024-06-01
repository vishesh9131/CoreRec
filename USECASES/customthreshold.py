
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import core_rec as cs
import vish_graphs as vg
from torch.utils.data import Dataset, DataLoader


# Load adjacency matrix
file_path = vg.generate_random_graph(40, seed=122)
adj_matrix = np.loadtxt(file_path, delimiter=",")
num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = len(adj_matrix[0])

#     # Convert adjacency matrix to dataset
graph_dataset = cs.GraphDataset(adj_matrix)
data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)

# Specify the node index for recommendation
node_index = 2
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)


# Predict recommendations using the simple neural network
# predictions = cs.predict(model, adj_matrix, node_index, top_k=5)
predictions = cs.predict(model, adj_matrix, node_index, top_k=5, threshold=0.1)
print(f"Recommended nodes for node {node_index}: {predictions}")

# #[18, 26, 7, 39] 0.9
# #[37, 26, 3, 39, 18] 0.8
# #[37, 11, 26, 9, 22] 0.4
# #[34, 18, 35, 37, 22] 0.1
