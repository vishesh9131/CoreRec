# import numpy as np
# import core_rec as cs
# import vish_graphs as vg
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# import random
# import os
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter


# def test_scores_for_different_node_counts(node_counts, seed=23, top_k=5, num_epochs=10):
#     jaccard_scores = []
#     adamic_adar_scores = []

#     for num_nodes in node_counts:
#         # Generate random graph
#         file_path = vg.generate_random_graph(num_nodes, seed=seed)
#         adj_matrix = np.loadtxt(file_path, delimiter=",")

#         # Convert adjacency matrix to dataset
#         graph_dataset = cs.GraphDataset(adj_matrix)
#         data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

#         # Define model parameters
#         num_layers = 2
#         d_model = 128
#         num_heads = 8
#         d_feedforward = 512
#         input_dim = len(adj_matrix[0])

#         # Initialize model, loss function, and optimizer
#         model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         # Train the model
#         cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

#         # Predict recommendations for a specific node
#         node_index = 2  # target node
#         recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=top_k)

#         # Calculate scores
#         aaj, aaj2 = cs.aaj_accuracy(adj_matrix, node_index, recommended_nodes)
#         jaccard_scores.append(aaj)
#         adamic_adar_scores.append(aaj2)

#     return jaccard_scores, adamic_adar_scores
 
# # Generate node counts from 20 to 200
# node_counts = range(20, 201, 20)

# # Test scores
# jaccard_scores, adamic_adar_scores = test_scores_for_different_node_counts(node_counts)

# # Function to format the y-axis ticks to show more decimal places
# def format_ticks(x, pos):
#     return f'{x:.2f}'  # Change '.2f' to however many decimal places you need

# # Assuming jaccard_scores and adamic_adar_scores are already computed and available
# fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # 3x2 grid of subplots

# # Line Plot
# axs[0, 0].plot(node_counts, jaccard_scores, label='Average Jaccard Score', marker='o')
# axs[0, 0].plot(node_counts, adamic_adar_scores, label='Average Adam/Adar Score', marker='x')
# axs[0, 0].set_title('Line Plot of Scores')
# axs[0, 0].set_xlabel('Number of Nodes')
# axs[0, 0].set_ylabel('Score')
# axs[0, 0].legend()
# axs[0, 0].grid(True)
# axs[0, 0].yaxis.set_major_formatter(FuncFormatter(format_ticks))

# # Bar Chart
# axs[0, 1].bar(node_counts, jaccard_scores, label='Average Jaccard Score', alpha=0.6)
# axs[0, 1].bar(node_counts, adamic_adar_scores, label='Average Adam/Adar Score', alpha=0.6)
# axs[0, 1].set_title('Bar Chart of Scores')
# axs[0, 1].set_xlabel('Number of Nodes')
# axs[0, 1].set_ylabel('Score')
# axs[0, 1].legend()
# axs[0, 1].grid(True)
# axs[0, 1].yaxis.set_major_formatter(FuncFormatter(format_ticks))

# # Box Plot
# data = [jaccard_scores, adamic_adar_scores]
# axs[1, 0].boxplot(data, labels=['Jaccard', 'Adam/Adar'])
# axs[1, 0].set_title('Box Plot of Scores')
# axs[1, 0].set_ylabel('Score')
# axs[1, 0].yaxis.set_major_formatter(FuncFormatter(format_ticks))

# # Histogram
# axs[1, 1].hist(jaccard_scores, bins=len(node_counts), alpha=0.7, label='Jaccard Scores')
# axs[1, 1].hist(adamic_adar_scores, bins=len(node_counts), alpha=0.7, label='Adam/Adar Scores')
# axs[1, 1].set_title('Histogram of Scores')
# axs[1, 1].set_xlabel('Score')
# axs[1, 1].set_ylabel('Frequency')
# axs[1, 1].legend()
# axs[1, 1].yaxis.set_major_formatter(FuncFormatter(format_ticks))

# # Area Plot
# axs[2, 0].fill_between(node_counts, jaccard_scores, label='Average Jaccard Score', step='mid', alpha=0.4)
# axs[2, 0].fill_between(node_counts, adamic_adar_scores, label='Average Adam/Adar Score', step='mid', alpha=0.4)
# axs[2, 0].set_title('Area Plot of Scores')
# axs[2, 0].set_xlabel('Number of Nodes')
# axs[2, 0].set_ylabel('Score')
# axs[2, 0].legend()
# axs[2, 0].grid(True)
# axs[2, 0].yaxis.set_major_formatter(FuncFormatter(format_ticks))

# # Adjust layout
# plt.tight_layout()
# plt.show()

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
from matplotlib.ticker import FuncFormatter

def test_scores_for_random_nodes(node_counts, num_random_nodes=5, top_k=5, num_epochs=10, seed=23):
    jaccard_scores = []
    adamic_adar_scores = []
    nodes_selected = []

    for num_nodes in node_counts:
        file_path = vg.generate_random_graph(num_nodes, seed=seed)
        adj_matrix = np.loadtxt(file_path, delimiter=",")
        graph_dataset = cs.GraphDataset(adj_matrix)
        data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)
        num_layers = 2
        d_model = 128
        num_heads = 8
        d_feedforward = 512
        input_dim = len(adj_matrix[0])
        model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

        for _ in range(num_random_nodes):
            node_index = random.randint(0, num_nodes - 1)
            recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=top_k)
            aaj, aaj2 = cs.aaj_accuracy(adj_matrix, node_index, recommended_nodes)
            jaccard_scores.append(aaj)
            adamic_adar_scores.append(aaj2)
            nodes_selected.append((num_nodes, node_index))

    return nodes_selected, jaccard_scores, adamic_adar_scores

# Generate node counts from 20 to 200
node_counts = range(20, 201, 20)

# Test scores
nodes_selected, jaccard_scores, adamic_adar_scores = test_scores_for_random_nodes(node_counts)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Scatter plot for Jaccard Scores
axs[0].scatter([n[0] for n in nodes_selected], jaccard_scores, c='blue', label='Jaccard Scores', alpha=0.6)
axs[0].set_title('Jaccard Scores for Random Nodes')
axs[0].set_xlabel('Number of Nodes')
axs[0].set_ylabel('Jaccard Score')
axs[0].grid(True)

# Scatter plot for Adam/Adar Scores
axs[1].scatter([n[0] for n in nodes_selected], adamic_adar_scores, c='red', label='Adam/Adar Scores', alpha=0.6)
axs[1].set_title('Adam/Adar Scores for Random Nodes')
axs[1].set_xlabel('Number of Nodes')
axs[1].set_ylabel('Adam/Adar Score')
axs[1].grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()