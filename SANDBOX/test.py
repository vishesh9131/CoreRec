# # ###############################################################################################################
# # #                            --testing_node_vishgraphs--                                                
# # # vish_graph module takes adjmatrix as input and has fns like                                                
# #     # 1. generate_random_graph(no_of_nodes,seed=23)
# #     # 2. find_top_nodes(adj_matrix) : greatest number of strong correlations or famous nodes top 5 
# #     # 3. draw_graph draws graph(matrix,set(range(len(adj_matrix))), set )
# # # note: just write 3d after draw_graph this will make it in xyz space
# # ###############################################################################################################
import numpy as np
# import vishgraph as vg
import core_rec as cs

# for trainig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

adj_matrix = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 1],
                       [1, 1, 0, 0],
                       [0, 1, 0, 0]])


# file_path = vg.generate_random_graph(32)
# adj_matrix = np.loadtxt(file_path, delimiter=",")
strong_relations, top_nodes = vg.find_top_nodes(adj_matrix)
vg.draw_graph_3d(adj_matrix, top_nodes)  # Pass both adj_matrix and top_nodes
print(top_nodes)







# for visulization and bipartite relationship
# file_path = vg.generate_random_graph(10)
# adj_matrix = np.loadtxt(file_path, delimiter=",")
# strong_relations, top_nodes = vg.find_top_nodes(adj_matrix)
# vg.draw_graph_3d_large(adj_matrix, top_nodes)
# print(top_nodes) 
# vg.draw_graph_3d(adj_matrix,top_nodes)
# adj_matrix1=vg.bipartite_matrix_maker(file_path)
# vg.show_bipartite_relationship(adj_matrix1)     
# vg.show_bipartite_relationship_with_cosine(adj_matrix)
