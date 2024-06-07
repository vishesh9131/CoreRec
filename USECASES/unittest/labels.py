import pandas as pd
import numpy as np
import sys
import os
import unittest
sys.path.append('/Users/visheshyadav/Documents/GitHub/CoreRec/vish_graphs')
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import networkx as nx
import core_rec as cs
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import vish_graphs as vg
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def draw_graph_3d(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges with optional weights
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
                if edge_weights is not None and i < len(edge_weights) and j < len(edge_weights[i]):
                    G[i][j]['weight'] = edge_weights[i][j]

    pos = nx.spring_layout(G, dim=3)  # Ensure pos is in 3D

    # Draw nodes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Chunking logic
    num_chunks = num_nodes // 1000 + 1
    nodes_list = list(G.nodes())
    chunk_legends = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * 1000
        end_idx = min((chunk_idx + 1) * 1000, num_nodes)
        chunk_nodes = nodes_list[start_idx:end_idx]

        for i in chunk_nodes:
            for j in chunk_nodes:
                if G.has_edge(i, j):
                    edge_alpha = 0.1 if transparent_labeled and (node_labels is None or i not in node_labels or j not in node_labels) else 1.0
                    edge_color = 'gray'
                    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], [pos[i][2], pos[j][2]], color=edge_color, alpha=edge_alpha)
                    
                    # Display edge weights if available
                    if 'weight' in G[i][j]:
                        mid_x = (pos[i][0] + pos[j][0]) / 2
                        mid_y = (pos[i][1] + pos[j][1]) / 2
                        mid_z = (pos[i][2] + pos[j][2]) / 2
                        ax.text(mid_x, mid_y, mid_z, str(G[i][j]['weight']), color='red', fontsize=8)

        for n in chunk_nodes:
            color = 'red' if top_nodes is not None and n in top_nodes else 'green' if recommended_nodes is not None and n in recommended_nodes else 'blue'
            node_alpha = 0.1 if transparent_labeled and (node_labels is None or n not in node_labels) else 1.0
            ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color, alpha=node_alpha)

            if node_labels is not None and n in node_labels:
                ax.text(pos[n][0], pos[n][1], pos[n][2], node_labels[n], fontsize=9)

    ax.text2D(0.95, 0.05, 'vishGraphs_use_in_labs', fontsize=8, color='gray', ha='right', va='bottom', transform=ax.transAxes)

    plt.title("3D Graph Visualization with Recommended Nodes Highlighted in Red and Top Nodes in Green")
    plt.show()
def find_top_nodes(matrix, num_nodes=3):
    relation_counts = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i, j] == matrix[j, i] == 1:
                relation_counts[i] += 1
                relation_counts[j] += 1
def generate_random_graph(num_people, file_path="graph_dataset.csv", seed=None):
    np.random.seed(seed)
    adj_matrix = np.zeros((num_people, num_people))

    for i in range(num_people):
        for j in range(i + 1, num_people):
            strength = np.random.rand()
            if strength < 0.1:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
            elif strength < 0.4:
                adj_matrix[i, j] = 1
            else:
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0

    np.savetxt(file_path, adj_matrix, delimiter=",")
    return file_path

class TestLabels(unittest.TestCase):
    def setUp(self):
        self.file_path = generate_random_graph(50, seed=122)
        self.adj_matrix = np.load





file_path = generate_random_graph(50, seed=122)
adj_matrix = np.loadtxt(file_path, delimiter=",")

# Read the CSV file into a DataFrame
df = pd.read_csv("/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/labelele.csv")

# # Find the top nodes
top_nodes = find_top_nodes(adj_matrix, num_nodes=5)

col = df.values
node_labels = {i: label for i, label in enumerate(col)}


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.file_path = generate_random_graph(50, seed=122)
        self.adj_matrix = np.loadtxt(self.file_path, delimiter=",")
        self.df = pd.read_csv("labelele.csv")
        self.col = self.df.values
        self.node_labels = {i: label for i, label in enumerate(self.col)}

    def test_labels(self):
        self.assertEqual(len(self.node_labels), len(self.col))

    def test_draw_graph(self):
        top_nodes = find_top_nodes(self.adj_matrix, num_nodes=5)
        draw_graph_3d(self.adj_matrix, top_nodes=top_nodes, node_labels=self.node_labels)
        self.assertTrue(True)  # If drawing completes without error, the test passes

if __name__ == '__main__':
    unittest.main()

