import numpy as np
import core_rec as cs
import vish_graphs as vg
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import os

# Generate random graph
file_path = vg.generate_random_graph(50, seed=23)
adj_matrix = np.loadtxt(file_path, delimiter=",")

file_path2 = vg.generate_weight_matrix(50,weight_range=(7,89), seed=23)
weight_matrix = np.loadtxt(file_path2, delimiter=",")


col=[]
for i in range(50):
    col.append(i)

node_labels = {i: label for i, label in enumerate(col)}

    # Convert adjacency matrix to dataset
graph_dataset = cs.GraphDataset(adj_matrix)
# graph_dataset = cs.GraphDataset(adj_matrix, weight_matrix)  # Include weight matrix
data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

    # Define model parameters
num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = len(adj_matrix[0])

    # Initialize model, loss function, and optimizer
model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
# model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim, use_weights=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=5)

    # Train the model
num_epochs = 10
cs.train_model(model, data_loader, criterion, optimizer, num_epochs)


    # Predict recommendations for a specific node
node_index = 2   #target node
recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=5, threshold=0.5)
print(f"Recommended nodes for node {node_index}: {recommended_nodes}")


    # Draw the graph in 3D
# vg.draw_graph_3d(adj_matrix, top_nodes, recommended_nodes,transparent_labeled=False,edge_weights=weight_matrix,node_labels=node_labels)


# print(jaccard)
aaj,aaj2 = cs.aaj_accuracy(adj_matrix,node_index,recommended_nodes)
print("Average Jaccard Score:",aaj)
print("Average Adam/Adar Score:",aaj2)

def test_scores_for_different_seeds(seeds, node_index=2, num_nodes=50, top_k=5, num_epochs=10):
    jaccard_scores = []
    adamic_adar_scores = []

    for seed in seeds:
        # Generate random graph
        file_path = vg.generate_random_graph(num_nodes, seed=seed)
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

        # Train the model
        cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

        # Predict recommendations for a specific node
        recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=top_k)

        # Calculate scores
        aaj, aaj2 = cs.aaj_accuracy(adj_matrix, node_index, recommended_nodes)
        jaccard_scores.append(aaj)
        adamic_adar_scores.append(aaj2)

    return jaccard_scores, adamic_adar_scores


# Generate 500 unique random seeds
seeds = random.sample(range(10000), 10)   # Example seeds: 10, 20, 30, 40, 50
# seeds = (10,102,241,234,123,324)   # Example seeds: 10, 20, 30, 40, 50


# Test scores
jaccard_scores, adamic_adar_scores = test_scores_for_different_seeds(seeds)



# # Function to save the model
# def save_model(model, path="model.pth"):
#     torch.save(model.state_dict(), path)

# Function to load the model
def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()


# Check if the model is already trained and saved
# model_path = "model.pth"
# if not os.path.exists(model_path):
#     # Train the model only if not already trained
#     cs.train_model(model, data_loader, criterion, optimizer, num_epochs)
#     save_model(model, model_path)
# else:
#     load_model(model, model_path)

# Modify the plotting section to use scatter plot
# plt.figure(figsize=(10, 5))
# plt.scatter(seeds, jaccard_scores, label='Average Jaccard Score', marker='o')
# plt.scatter(seeds, adamic_adar_scores, label='Average Adam/Adar Score', marker='x')
# plt.xlabel('Seed')
# plt.ylabel('Score')
# plt.title('Average Jaccard and Adam/Adar Scores for Different Seeds')
# plt.legend()
# plt.grid(True)
# plt.show()

fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid of subplots

# Line Plot
axs[0, 0].plot(seeds, jaccard_scores, label='Average Jaccard Score', marker='o')
axs[0, 0].plot(seeds, adamic_adar_scores, label='Average Adam/Adar Score', marker='x')
axs[0, 0].set_title('Line Plot of Scores')
axs[0, 0].set_xlabel('Seed')
axs[0, 0].set_ylabel('Score')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Bar Chart
axs[0, 1].bar(seeds, jaccard_scores, label='Average Jaccard Score', alpha=0.6)
axs[0, 1].bar(seeds, adamic_adar_scores, label='Average Adam/Adar Score', alpha=0.6)
axs[0, 1].set_title('Bar Chart of Scores')
axs[0, 1].set_xlabel('Seed')
axs[0, 1].set_ylabel('Score')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Box Plot
data = [jaccard_scores, adamic_adar_scores]
axs[1, 0].boxplot(data, labels=['Jaccard', 'Adam/Adar'])
axs[1, 0].set_title('Box Plot of Scores')
axs[1, 0].set_ylabel('Score')

# Histogram
axs[1, 1].hist(jaccard_scores, bins=5, alpha=0.7, label='Jaccard Scores')
axs[1, 1].hist(adamic_adar_scores, bins=5, alpha=0.7, label='Adam/Adar Scores')
axs[1, 1].set_title('Histogram of Scores')
axs[1, 1].set_xlabel('Score')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].legend()

# Area Plot
axs[2, 0].fill_between(seeds, jaccard_scores, label='Average Jaccard Score', step='mid', alpha=0.4)
axs[2, 0].fill_between(seeds, adamic_adar_scores, label='Average Adam/Adar Score', step='mid', alpha=0.4)
axs[2, 0].set_title('Area Plot of Scores')
axs[2, 0].set_xlabel('Seed')
axs[2, 0].set_ylabel('Score')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()