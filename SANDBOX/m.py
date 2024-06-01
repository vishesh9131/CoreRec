import vish_graphs as vg
import numpy as np


file_path = vg.generate_random_graph(30, seed=122)
adj_matrix = np.loadtxt(file_path, delimiter=",")
top_nodes=vg.find_top_nodes(adj_matrix)

# node_labels = {i: f"Node {i}" for i in range(num)}
node_labels="123"

# 2d me draw ke liye
# vg.draw_graph(adj_matrix,top_nodes)

# 3d  me draw ke liye
vg.draw_graph(adj_matrix,top_nodes,node_labels=node_labels)
