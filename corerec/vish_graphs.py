
# ###############################################################################################################
#                                             --vishgraphs--                                                  
# vish_graph module takes adjmatrix as input and provides various functionalities for graph manipulation and
# visualization. It includes functions for:
#     1. Generating random graphs with optional seeding (generate_random_graph)
#     2. Generating random weight matrices for graphs (generate_weight_matrix)
#     3. Finding top nodes based on the number of strong correlations (find_top_nodes)
#     4. Exporting graph data and node labels to a CSV file (export_graph_data_to_csv)
#     5. Drawing 2D and 3D graph visualizations with support for node coloring, transparency, and edge weights
#        (draw_graph, draw_graph_3d)
#     6. Visualizing bipartite relationships and community detection based on cosine similarity
#        (show_bipartite_relationship, show_bipartite_relationship_with_cosine)
#     7. scale_and_save_matrices feature to scale given adj
# Note: Append '3d' to draw_graph (draw_graph_3d) for 3D visualizations in xyz space.
# ###############################################################################################################

import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities
import corerec.core_rec as cs
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from scipy.sparse import lil_matrix
import scipy.sparse as sp
from memory_profiler import profile
from scipy.sparse import dok_matrix
import multiprocessing
from functools import partial
from corerec.optimal_path.optimal_path import *

def generate_random_graph(num_people, file_path="graph_dataset.csv", seed=None):
    np.random.seed(seed)
    adj_matrix = np.zeros((num_people, num_people))

    # Wrap the outer loop with tqdm for a progress bar
    for i in tqdm(range(num_people), desc="Generating graph"):
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

# def generate_random_graph(num_people, file_path="graph_dataset.csv", seed=None):
#     np.random.seed(seed)
#     adj_matrix = np.zeros((num_people, num_people))

#     for i in range(num_people):
#         for j in range(i + 1, num_people):
#             strength = np.random.rand()
#             if strength < 0.1:
#                 adj_matrix[i, j] = 1
#                 adj_matrix[j, i] = 1
#             elif strength < 0.4:
#                 adj_matrix[i, j] = 1
#             else:
#                 adj_matrix[i, j] = 0
#                 adj_matrix[j, i] = 0

#     np.savetxt(file_path, adj_matrix, delimiter=",")
#     return file_path

def generate_connections(num_people, i, p_strong, p_weak):
    num_connections = num_people - i - 1
    if num_connections <= 0:
        return None, None
    connections = np.random.rand(num_connections)
    strong_connections = np.where(connections < p_strong)[0] + i + 1
    weak_connections = np.where((connections >= p_strong) & (connections < p_strong + p_weak))[0] + i + 1
    return strong_connections, weak_connections

@profile
def generate_large_random_graph(num_people, file_path="large_random_graph.csv", seed=None):
    np.random.seed(seed)
    adj_matrix = lil_matrix((num_people, num_people), dtype=np.int8)

    p_strong = 0.1
    p_weak = 0.3

    # Setup multiprocessing pool within the main guard
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(generate_connections, num_people, p_strong=p_strong, p_weak=p_weak)
    results = pool.map(func, range(num_people))
    pool.close()
    pool.join()

    progress_bar = tqdm(total=num_people, desc="Generating Graph", unit="rows")

    for i, result in enumerate(results):
        if result[0] is not None:
            strong_connections, weak_connections = result
            adj_matrix[i, strong_connections] = 1
            adj_matrix[strong_connections, i] = 1
            adj_matrix[i, weak_connections] = 1
        progress_bar.update(1)

    progress_bar.close()

    # Convert to a dense matrix and save as CSV
    adj_matrix_dense = adj_matrix.toarray()
    np.savetxt(file_path, adj_matrix_dense, delimiter=",")

    return file_path

def generate_weight_matrix(num_nodes, weight_range=(1, 10), file_path="weight_matrix.csv", seed=None):
    """
    Generates a random weight matrix for a given number of nodes.
    """
    if seed is not None:
        np.random.seed(seed)

    weight_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = np.random.randint(weight_range[0], weight_range[1] + 1)
            weight_matrix[i, j] = weight_matrix[j, i] = weight

    np.savetxt(file_path, weight_matrix, delimiter=",")
    return file_path

def find_top_nodes(matrix, num_nodes=3):
    relation_counts = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i, j] == matrix[j, i] == 1:
                relation_counts[i] += 1
                relation_counts[j] += 1
    
    top_nodes = sorted(range(len(relation_counts)), key=lambda i: relation_counts[i], reverse=True)[:num_nodes]
    print(f"The top {num_nodes} nodes with the greatest number of strong correlations are: {top_nodes}")
    return top_nodes

def export_graph_data_to_csv(adj_matrix, node_labels, csv_file):
    """
    Export graph data and node labels to a CSV file using the csv module.

    Args:
    adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
    node_labels (dict): A dictionary with node indices as keys and labels as values.
    csv_file (str): Path to the output CSV file.
    """
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write headers
        headers = [f'Node {i}' for i in range(len(adj_matrix))]
        headers.append('Label')
        writer.writerow(headers)
    
        # Write data rows
        for index, row in enumerate(adj_matrix):
            # Append the label to the row
            row_with_label = list(row) + [node_labels.get(index, '')]
            writer.writerow(row_with_label)

def draw_graph(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None):
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

    pos = nx.spring_layout(G)  # 2D position layout

    # Draw nodes with color coding
    node_colors = []
    for node in G.nodes():
        if recommended_nodes is not None and node in recommended_nodes:
            node_colors.append('green')
        elif top_nodes is not None and node in top_nodes:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # Draw edges and optionally display weights
    for i, j in G.edges():
        edge_color = 'gray'
        edge_alpha = 0.1 if transparent_labeled and (node_labels is None or i not in node_labels or j not in node_labels) else 0.5
        nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=1.0, alpha=edge_alpha, edge_color=edge_color)
        
        if 'weight' in G[i][j]:
            mid_x = (pos[i][0] + pos[j][0]) / 2
            mid_y = (pos[i][1] + pos[j][1]) / 2
            plt.text(mid_x, mid_y, str(G[i][j]['weight']), color='red', fontsize=8)

    # Draw labels
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
    else:
        nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Graph Visualization with Recommended Nodes Highlighted in Green and Top Nodes in Red")
    plt.show()

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


def show_bipartite_relationship(adj_matrix):
    B = nx.Graph()

    num_nodes = len(adj_matrix)
    B.add_nodes_from(range(num_nodes), bipartite=0)
    B.add_nodes_from(range(num_nodes, 2*num_nodes), bipartite=1)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                B.add_edge(i, j + num_nodes)

    pos = nx.bipartite_layout(B, nodes=range(num_nodes))
    nx.draw(B, pos, with_labels=True, node_size=500, node_color='skyblue')
    plt.title("Bipartite Relationship Visualization")
    plt.show()

def show_bipartite_relationship_with_cosine(adj_matrix):
    num_nodes = len(adj_matrix)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(adj_matrix)

    # Create bipartite graph based on cosine similarity
    B = nx.Graph()
    B.add_nodes_from(range(num_nodes), bipartite=0)
    B.add_nodes_from(range(num_nodes, 2*num_nodes), bipartite=1)

    # Add edges based on cosine similarity
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and cosine_sim[i][j] > 0:  # Only consider positive similarity
                B.add_edge(i, j + num_nodes, weight=cosine_sim[i][j])

    # Detect communities using a community detection algorithm
    communities = list(greedy_modularity_communities(B))

    # Create a color map for the communities
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = i

    # Draw the bipartite graph with communities
    pos = nx.bipartite_layout(B, nodes=range(num_nodes))
    node_colors = [color_map.get(node, 0) for node in B.nodes()]

    nx.draw(B, pos, with_labels=True, node_size=500, node_color=node_colors, cmap=plt.cm.rainbow)
    plt.title("Bipartite Relationship Visualization with Cosine Similarity-based Communities")
    plt.show()


def bipartite_matrix_maker(csv_path):
    adj_matrix = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            values = [float(value) for value in row]
            adj_matrix.append(values)
    return adj_matrix

    # Convert to a sparse CSR matrix if not already in that format
    if not isinstance(adj_matrix, csr_matrix):
        sparse_adj_matrix = csr_matrix(adj_matrix)
    else:
        sparse_adj_matrix = adj_matrix
    
    # Initialize the graph from the sparse matrix
    G = nx.convert_matrix.from_scipy_sparse_array(sparse_adj_matrix)

    # Use a more efficient layout algorithm for large graphs
    print("Preparing graph layout...")
    if G.number_of_nodes() > 1000:
        pos = {}
        iterations = 10
        with tqdm(total=iterations, desc="Computing layout") as pbar:
            for _ in range(iterations):
                pos = nx.spring_layout(G, pos=pos, iterations=1)  # Update position incrementally
                pbar.update(1)
    else:
        pos = nx.kamada_kawai_layout(G)  # Default settings for smaller graphs

    # Prepare node colors efficiently
    node_colors = np.full(G.number_of_nodes(), 'skyblue')
    if recommended_nodes:
        node_colors[recommended_nodes] = 'green'
    if top_nodes:
        node_colors[top_nodes] = 'red'

    # Drawing the graph
    print("Drawing graph...")
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    if node_labels:
        labels = {node: node_labels[node] for node in node_labels if node in pos}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=5)

    plt.title("Large Graph Visualization")
    plt.show()


    # Convert to a sparse CSR matrix if not already in that format
    if not isinstance(adj_matrix, csr_matrix):
        sparse_adj_matrix = csr_matrix(adj_matrix)
    else:
        sparse_adj_matrix = adj_matrix
    
    # Initialize the graph from the sparse matrix
    G = nx.convert_matrix.from_scipy_sparse_array(sparse_adj_matrix)

    # Use a more efficient layout algorithm for large graphs
    print("Preparing graph layout...")
    if G.number_of_nodes() > 1000:
        pos = nx.spring_layout(G, scale=2)  # Faster for large graphs
    else:
        pos = nx.kamada_kawai_layout(G, scale=2)  # Better for smaller or medium-sized graphs

    # Prepare node colors
    node_colors = ['skyblue'] * G.number_of_nodes()
    if recommended_nodes:
        for node in tqdm(recommended_nodes, desc="Coloring recommended nodes"):
            node_colors[node] = 'green'
    if top_nodes:
        for node in tqdm(top_nodes, desc="Coloring top nodes"):
            node_colors[node] = 'red'

    # Drawing the graph
    print("Drawing graph...")
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5)

    plt.title("Large Graph Visualization")
    plt.show()



def scale_and_save_matrices(input_file, output_dir, num_matrices):
    # Load the existing matrix
    matrix = np.genfromtxt(input_file, delimiter=',')

    # Create and save the scaled matrices
    for i in range(1, num_matrices + 1):
        scaled_matrix = matrix * i  # Scale the matrix by a factor of i
        np.savetxt(f'{output_dir}/label_{i}.csv', scaled_matrix, delimiter=',', fmt='%d')

# List of possible misspellings
possible_misspellings = [
    "scale_and_save_matrix", "scale_n_save_matrices", "scal_and_save_matrices", 
    "scale_and_save_matrces", "scale_and_sav_matrices", "scale_n_save_matrix",
    "scaleandsavematrices", "scaleandsave_matrix", "scale_n_save_matricies",
    "scaling_and_saving_matrices", "scale_and_save_matricies", "scale_save_matrices",
    "scale_save_matrix", "scale_save_matricies", "scal_save_matrices",
    "scaling_save_matrices", "scaling_save_matrix", "scale_saving_matrices",
    "scale_and_saving_matrices", "scale_saving_matrix"
]

# Creating aliases for the function
for misspelling in possible_misspellings:
    globals()[misspelling] = scale_and_save_matrices


# Example usage
# scale_and_save_matrices('SANDBOX/label.csv', 'SANDBOX/delete', 10)
graph = []
start_city = 0

def run_optimal_path():
    show_path(graph, start_city)
