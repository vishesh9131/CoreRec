# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from engine.core_rec import GraphTransformer, GraphDataset, train_model, predict

# # Constants
# NUM_NODES = 30_000_000
# BATCH_SIZE = 1024
# NUM_EPOCHS = 10
# NUM_LAYERS = 2
# D_MODEL = 128
# NUM_HEADS = 8
# D_FEEDFORWARD = 512
# LEARNING_RATE = 0.001
# TARGET_NODE_INDEX = 12453

# # Load the graph data from CSV
# def load_graph_data(filename='SANDBOX/Analysis/data_mother/large_network.csv'):
#     df = pd.read_csv(filename)
#     adj_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
#     for _, row in df.iterrows():
#         source, target, weight = int(row['source']), int(row['target']), float(row['weight'])
#         adj_matrix[source, target] = weight
#         adj_matrix[target, source] = weight  # Assuming undirected graph
#     return adj_matrix

# # Main function to train the model and get recommendations
# def main():
#     # Load graph data
#     adj_matrix = load_graph_data()

#     # Create GraphDataset and DataLoader
#     graph_dataset = GraphDataset(adj_matrix)
#     data_loader = DataLoader(graph_dataset, batch_size=BATCH_SIZE, shuffle=True)

#     # Initialize model, loss function, and optimizer
#     input_dim = adj_matrix.shape[1]
#     model = GraphTransformer(NUM_LAYERS, D_MODEL, NUM_HEADS, D_FEEDFORWARD, input_dim)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # Train the model
#     train_model(model, data_loader, criterion, optimizer, NUM_EPOCHS)

#     # Predict recommendations for the target node
#     recommended_nodes = predict(model, adj_matrix, TARGET_NODE_INDEX, top_k=5, threshold=0.5)
#     print(f"Recommended nodes for node {TARGET_NODE_INDEX}: {recommended_nodes}")

# if __name__ == "__main__":
#     main()



import numpy as np
import core_rec as cs
import vish_graphs as vg
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



# Generate random graph
file_path = pd.read_csv("SANDBOX/Analysis/data_mother/large_network.csv")
adj_matrix = np.loadtxt(file_path, delimiter=",")

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
num_epochs = 2
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)


    # Predict recommendations for a specific node
node_index = 12453   #target node
recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=5, threshold=0.5)
print(f"Recommended nodes for node {node_index}: {recommended_nodes}")

