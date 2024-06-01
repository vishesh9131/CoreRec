#######################################################################################################
#                                              COREREC X VISHGRAPHS
#######################################################################################################
#                                               SOME USE CASES 
#######################################################################################################
'''
problem statement :
Q.1) Generate a random dataset of graph(nodes=72) 
and visulize it 2d and 3d (mandatory to use seed=332)

Q.2) print its adj matrix and find Top nodes.
(mandatory to use seed=332)

Q.3) Generate a Bipartite Graph of that dataset 
visulize it with cosine similarity.

Q.4) Recommend Nodes to Node 7 and visulize it.
'''

import vish_graphs as vg
import core_rec as cs
from common_import import * 
#######################################################################################################
# Q1
# file_path = vg.generate_random_graph(72, seed=332)
# adj_matrix = np.loadtxt(file_path, delimiter=",")
# # vg.draw_graph(adj_matrix)
# vg.draw_graph_3d(adj_matrix, None, None)
#######################################################################################################
# Q2
# file_path = vg.generate_random_graph(72, seed=332)
# adj_matrix=np.loadtxt(file_path,delimiter=",")
# # print("The Adj matrix is :",adj_matrix)

# top_nodes=vg.find_top_nodes(adj_matrix)
# vg.draw_graph(adj_matrix,top_nodes=top_nodes)
#######################################################################################################
# Q3  Generate a Bipartite Graph of that dataset 
#     visulize it with cosine similarity.
# file_path = vg.generate_random_graph(72, seed=332)
# matrix=vg.bipartite_matrix_maker(file_path)
# vg.show_bipartite_relationship_with_cosine(matrix)
#######################################################################################################
# # Q.4) Recommend Nodes to Node 7 and visulize it.
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# # in 5 steps 
# # 1.Generate random graph and load adjacency matrix
# # 2. Initialize Transformer Model
# # 3. Define your loss function, optimizer, and other training parameters
# # 4. Train the model
# # 5. Use the trained model for node recommendations
# # 6. visulize it

# # 1
# file_path = vg.generate_random_graph(40,seed=122)
# adj_matrix = np.loadtxt(file_path, delimiter=",")
# top_nodes = vg.find_top_nodes(adj_matrix)
# # vg.draw_graph(adj_matrix,top_nodes=top_nodes)

# # 2
# num_layers = 3
# d_model = 128
# num_heads = 4
# d_feedforward = 256
# input_dim = adj_matrix.shape[0] # Input dimension should match the number of nodes in the graph
# model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)

# dataset = cs.GraphDataset(adj_matrix)
# data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# # 3
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 150

# # 4
# cs.train_model(model, data_loader, criterion, optimizer, num_epochs)

# # 5
# node_index = 2
# predictions = cs.predict(model, adj_matrix, node_index, top_k=5)
# print(f"Recommended nodes for node {node_index}: {predictions}")
# print("Popular Nodes are :",top_nodes)

# # 6
# vg.draw_graph_3d(adj_matrix,top_nodes=top_nodes,recommended_nodes=predictions)
# top nodes are pushed in predictions list
#######################################################################################################
