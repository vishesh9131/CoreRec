import vish_graphs as vg
import core_rec as cr 
import numpy as np

file_path = vg.generate_random_graph(1000,file_path="swad.csv",seed=123)
adj_matrix = np.loadtxt(file_path, delimiter=",")
top_nodes= vg.find_top_nodes(adj_matrix,num_nodes=5)

