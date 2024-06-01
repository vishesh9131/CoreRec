# Importing core_rec as cr and vish_graphs as vg
import core_rec as cr
import vish_graphs as vg
import numpy as np
import networkx as nx
import torch

num_people=100
file_path = vg.generate_random_graph(num_people, file_path="blobbag.csv",seed=56)
    # adj_matrix= np.loadtxt(file_path,delimiter=",")
file_path2 = vg.generate_weight_matrix(12,weight_range=(0,1),file_path="weight_matrix.csv",seed=89)
weight_matrix=np.loadtxt(file_path2,delimiter=",")
graph_data=np.loadtxt(file_path,delimiter=",")
top_nodes=vg.find_top_nodes(adj_matrix,num_nodes=2)



label=[]
for i in range(num_people):
    i+=1
    label.append(i)

node_labels = {i: label for i, label in enumerate(label)}


# convert matrix to eligible format for feeding into model
graph_dataset = cr.GraphDataset(adj_matrix)
data_loader = cr.DataLoader(graph_dataset, batch_size=5, shuffle=True)

# definenig model properties
num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = len(adj_matrix[0])

# init model 
model = cr.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)

# definenig training properties
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# init train
cr.train_model(model, data_loader, criterion, optimizer, num_epochs)

# defining predictors properties 
node_index = 10


# Print the recommendations and explanations
print("Recommended Nodes:", recommended_indices)
for explanation in explanations:
    print("Node:", explanation["node"])
    print("Score:", explanation["score"])
    print("Jaccard Similarity:", explanation["jaccard_similarity"])
    print("Adamic/Adar Index:", explanation["adamic_adar_index"])
    print("Explanation:", explanation["explanation"])