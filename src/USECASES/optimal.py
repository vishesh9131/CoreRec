import corerec.vish_graphs as vg
import numpy as np
import pandas as pd


# file_path = vg.generate_random_graph(50, seed=122)
# adj_matrix = np.loadtxt(file_path, delimiter=",")

vg.graph = [[0, 10, 15, 20], [5, 0, 9, 10], [6, 13, 0, 12], [8, 8, 9, 0]]
vg.start_city = 0


# Call the function to run the optimal path visualization
vg.run_optimal_path()
