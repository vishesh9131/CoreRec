# def draw_graph(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None):
#     G = nx.Graph()
#     num_nodes = adj_matrix.shape[0]

#     # Add nodes
#     for i in range(num_nodes):
#         G.add_node(i)

#     # Add edges with optional weights
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adj_matrix[i, j] == 1:
#                 weight = edge_weights[i, j] if edge_weights is not None else 1
#                 G.add_edge(i, j, weight=weight)

#     pos = nx.spring_layout(G)  # 2D position layout

#     # Draw nodes with color coding
#     node_colors = []
#     for node in G.nodes():
#         if recommended_nodes is not None and node in recommended_nodes:
#             node_colors.append('green')
#         elif top_nodes is not None and node in top_nodes:
#             node_colors.append('red')
#         else:
#             node_colors.append('skyblue')

#     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

#     # Draw edges and optionally display weights
#     for i, j in G.edges():
#         edge_color = 'gray'
#         edge_alpha = 0.1 if transparent_labeled and (node_labels is None or i not in node_labels or j not in node_labels) else 0.5
#         nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=1.0, alpha=edge_alpha, edge_color=edge_color)
        
#         if edge_weights is not None:
#             mid_x = (pos[i][0] + pos[j][0]) / 2
#             mid_y = (pos[i][1] + pos[j][1]) / 2
#             plt.text(mid_x, mid_y, str(edge_weights[i, j]), color='red', fontsize=8)

#     # Draw labels
#     if node_labels is not None:
#         nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
#     else:
#         nx.draw_networkx_labels(G, pos, font_size=12)

#     plt.title("Graph Visualization with Recommended Nodes Highlighted in Green and Top Nodes in Red")
#     plt.show()

# def draw_graph_3d(adj_matrix, top_nodes=None, recommended_nodes=None, node_labels=None, transparent_labeled=True, edge_weights=None):
#     G = nx.Graph()
#     num_nodes = adj_matrix.shape[0]

#     # Add nodes
#     for i in range(num_nodes):
#         G.add_node(i)

#     # Add edges with optional weights
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adj_matrix[i, j] == 1:
#                 weight = edge_weights[i, j] if edge_weights is not None else 1
#                 G.add_edge(i, j, weight=weight)

#     pos = nx.spring_layout(G, dim=3)  # Ensure pos is in 3D

#     # Draw nodes
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Chunking logic
#     num_chunks = num_nodes // 1000 + 1
#     nodes_list = list(G.nodes())
#     chunk_legends = []

#     for chunk_idx in range(num_chunks):
#         start_idx = chunk_idx * 1000
#         end_idx = min((chunk_idx + 1) * 1000, num_nodes)
#         chunk_nodes = nodes_list[start_idx:end_idx]

#         for i in chunk_nodes:
#             for j in chunk_nodes:
#                 if G.has_edge(i, j):
#                     edge_alpha = 0.1 if transparent_labeled and (node_labels is None or i not in node_labels or j not in node_labels) else 1.0
#                     edge_color = 'gray'
#                     ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], [pos[i][2], pos[j][2]], color=edge_color, alpha=edge_alpha)
                    
#                     # Display edge weights
#                     if edge_weights is not None:
#                         mid_x = (pos[i][0] + pos[j][0]) / 2
#                         mid_y = (pos[i][1] + pos[j][1]) / 2
#                         mid_z = (pos[i][2] + pos[j][2]) / 2
#                         ax.text(mid_x, mid_y, mid_z, str(edge_weights[i, j]), color='red', fontsize=8)

#         for n in chunk_nodes:
#             color = 'red' if top_nodes is not None and n in top_nodes else 'green' if recommended_nodes is not None and n in recommended_nodes else 'blue'
#             node_alpha = 0.1 if transparent_labeled and (node_labels is None or n not in node_labels) else 1.0
#             ax.scatter(pos[n][0], pos[n][1], pos[n][2], color=color, alpha=node_alpha)

#             if node_labels is not None and n in node_labels:
#                 ax.text(pos[n][0], pos[n][1], pos[n][2], node_labels[n], fontsize=9)

#     ax.text2D(0.95, 0.05, 'vishGraphs_use_in_labs', fontsize=8, color='gray', ha='right', va='bottom', transform=ax.transAxes)

#     plt.title("3D Graph Visualization with Recommended Nodes Highlighted in Red and Top Nodes in Green")
#     plt.show()
