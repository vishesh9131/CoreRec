import vish_graphs as vg
import numpy as np
import pandas as pd

# numpeople means nodes and num_nodes means weights
# they have to be identical 
num_people=30
num_nodes=20

# generating random graph and weight matrix simple to hai....
file_path=vg.generate_random_graph(num_people,file_path="graph.csv", seed=42)
file_path1=vg.generate_weight_matrix(num_nodes, weight_range=(1, 10), file_path="weight_matrix.csv", seed=42)

# wo generated data(csv) ko yaha load karinge 
adj_matrix = np.loadtxt(file_path, delimiter=",")
weight_matrix = np.loadtxt(file_path1, delimiter=",")

# yeh to pata hi hai topnodes kesy bnana ha
top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=10)

# nl=node ke labels 
# node_labels yaha pedict banra hai aise :- {0:1,1:2,2:3,3:4,4:5,5:6}
nl = [1,2,3,4,5,6]
node_labels = {i: label for i, label in enumerate(nl)}

# visulization part my favorite...
# vg.draw_graph(adj_matrix, node_labels=node_labels,top_nodes=top_nodes,edge_weights=weight_matrix)

# 3dVisulizn
vg.draw_graph_3d(adj_matrix, node_labels=node_labels,top_nodes=top_nodes,edge_weights=weight_matrix)
