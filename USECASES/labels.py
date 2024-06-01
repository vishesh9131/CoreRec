
import core_rec as cs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import core_rec as cs
import vish_graphs as vg
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
import vish_graphs as vg

import pandas as pd
import numpy as np
file_path = vg.generate_random_graph(50, seed=122)
adj_matrix = np.loadtxt(file_path, delimiter=",")

# Read the CSV file into a DataFrame
df = pd.read_csv("labelele.csv")

# # Find the top nodes
top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=5)

# # Define node labels
# node_labels = [1,2,3,4,5,6,7,8,9,10] #custom labels
# {i: f"Node {i}" for i in range(num_people)} #labels in a itr
col = df.values
node_labels = {i: label for i, label in enumerate(col)}

# # Visualize the 3D graph with labels
vg.draw_graph_3d(adj_matrix, top_nodes=top_nodes, node_labels=node_labels)

# # Visualize the 3D graph with labels
# vg.draw_graph_3d(adj_matrix, top_nodes=top_nodes,node_labels=node_labels)

##########################################

# # #IF YOU HAVE CSV FILE AS A LABELER
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import core_rec as cs
# import vish_graphs as vg
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd 
# import numpy as np
# import vish_graphs as vg
# import numpy as np
# file_path = vg.generate_random_graph(10, seed=122)

# adj_matrix = np.loadtxt(file_path, delimiter=",")

# # Read the CSV file into a DataFrame
# df = pd.read_csv("labelele.csv")

# # Access the column and convert it to a NumPy array
# col = df.values

# # Convert NumPy array to dictionary
# node_labels = {i: label for i, label in enumerate(col)}

# # Find the top nodes
# top_nodes = vg.find_top_nodes(adj_matrix, num_nodes=5)

# # Visualize the 2D graph with labels
# # vg.draw_graph(adj_matrix, top_nodes=top_nodes, node_labels=node_labels)

# # export in csv 
# vg.export_graph_data_to_csv(adj_matrix, node_labels, "output_graph_data.csv")