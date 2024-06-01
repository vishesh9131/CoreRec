# # ###############################################################################################################
# # #                                             --vishgraphs--                                                  
# # # vish_graph module takes adjmatrix as input and has fns like                                                
# #     # 1. generate_random_graph(no_of_nodes,seed=23)
# #     # 2. find_top_nodes(adj_matrix) : greatest number of strong correlations or famous nodes top 5 
# #     # 3. draw_graph draws graph(matrix,set(range(len(adj_matrix))), set )
# # # note: just write 3d after draw_graph this will make it in xyz space
# # ###############################################################################################################
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms.community import greedy_modularity_communities
import core_rec as cs

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

def find_top_nodes(matrix, num_nodes=10):
    relation_counts = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix[i])):
            if matrix[i, j] == matrix[j, i] == 1:
                relation_counts[i] += 1
                relation_counts[j] += 1
    
    top_nodes = sorted(range(len(relation_counts)), key=lambda i: relation_counts[i], reverse=True)[:num_nodes]
    print(f"The top {num_nodes} nodes with the greatest number of strong correlations are: {top_nodes}")
    return top_nodes


# def draw_graph(adj_matrix, top_nodes=None, recommended_nodes=None):
#     G = nx.Graph()
#     num_nodes = adj_matrix.shape[0]

#     # Add nodes
#     for i in range(num_nodes):
#         G.add_node(i)

#     # Add edges
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             if adj_matrix[i, j] == 1:
#                 G.add_edge(i, j)

#     pos = nx.spring_layout(G)

#     # Draw nodes
#     node_colors = []
#     for node in G.nodes():
#         if recommended_nodes is not None and node in recommended_nodes:
#             node_colors.append('green')
#         elif top_nodes is not None and node in top_nodes:
#             node_colors.append('red')
#         else:
#             node_colors.append('skyblue')

#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

#     # Draw edges
#     nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=12)

#     plt.title("Graph Visualization with Recommended Nodes Highlighted in Green and Top Nodes in Red")
#     plt.show()

# def draw_graph_3d(adj_matrix, node_index, top_nodes):
#     nodes = set(range(len(adj_matrix)))
#     fig = plt.figure(figsize=(12, 8))
#     start_time = time.time()
#     ax = fig.add_subplot(111, projection='3d')

#     pos = np.random.rand(len(nodes), 3)

#     num_chunks = len(nodes) // 1000 + 1
#     nodes_list = list(nodes)
#     chunk_legends = []

#     for chunk_idx in range(num_chunks):
#         start_idx = chunk_idx * 1000
#         end_idx = min((chunk_idx + 1) * 1000, len(nodes))
#         chunk_nodes = nodes_list[start_idx:end_idx]

#         for i in range(len(adj_matrix)):
#             for j in range(len(adj_matrix[i])):
#                 if adj_matrix[i, j] == 1 and i in chunk_nodes and j in chunk_nodes:
#                     ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]], 'gray')

#         for n in chunk_nodes:
#             color = 'red' if n == node_index else 'blue' if n in top_nodes else 'black'
#             ax.scatter(pos[n, 0], pos[n, 1], pos[n, 2], color=color)

#     ax.text(0.95, 0.05, 0.05, 'vishGraphs_use_in_labs', fontsize=8, color='gray', ha='right', va='bottom', transform=ax.transAxes)
#     if num_chunks > 1:
#         ax.legend(chunk_legends, title='Chunks', loc='upper left')

#     plt.show()

#     elapsed_time = time.time() - start_time
#     print(f"Time taken to process the graph: {elapsed_time:.2f} seconds")

# def draw_graph_3d(adj_matrix, top_nodes=None, recommended_nodes=None):
#     G = nx.Graph()
#     num_nodes = adj_matrix.shape[0]

#     # Add nodes
#     for i in range(num_nodes):
#         G.add_node(i)

#     # Add edges
#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             if adj_matrix[i, j] == 1:
#                 G.add_edge(i, j)

#     pos = nx.spring_layout(G, dim=3)  # Ensure pos is in 3D

#     # Draw nodes
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     for i in range(num_nodes):
#         for j in range(i+1, num_nodes):
#             if adj_matrix[i, j] == 1:
#                 ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], [pos[i][2], pos[j][2]], 'gray')
#     if top_nodes is not None:
#         for n in G.nodes():
#             color = 'red' if n in recommended_nodes else 'green' if n in top_nodes else 'blue'
#             ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color)
#     else:
#         for n in G.nodes():
#             color = 'blue' 
#             ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color)

#     ax.text(0.95, 0.05, 0.05, 'vishGraphs_use_in_labs', fontsize=8, color='gray', ha='right', va='bottom', transform=ax.transAxes)

#     plt.title("3D Graph Visualization with Recommended Nodes Highlighted in Red and Top Nodes in Green")
#     plt.show()


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

def draw_graph(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if recommended_nodes is not None and node in recommended_nodes:
            node_colors.append('green')
        elif top_nodes is not None and node in top_nodes:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw labels
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
    else:
        nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Graph Visualization with Recommended Nodes Highlighted in Green and Top Nodes in Red")
    plt.show()
def draw_graph_3d(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G, dim=3)  # Ensure pos is in 3D

    # Draw nodes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], [pos[i][2], pos[j][2]], color='gray')

    if top_nodes is not None:
        for n in G.nodes():
            color = 'red' if n in top_nodes else 'blue'
            ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color)
    elif recommended_nodes is not None:
        for n in G.nodes():
            color = 'green' if n in recommended_nodes else 'blue'
            ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color)
    else:
        for n in G.nodes():
            color = 'blue'
            ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color)

    if node_labels is not None:
        for n in G.nodes():
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