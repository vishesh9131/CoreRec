import numpy as np
import vish_graphs as vg 
import core_rec as cr 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


num_people=100
file_path = vg.generate_random_graph(num_people, file_path="blobbag.csv",seed=56)
    # adj_matrix= np.loadtxt(file_path,delimiter=",")
file_path2 = vg.generate_weight_matrix(12,weight_range=(0,1),file_path="weight_matrix.csv",seed=89)
weight_matrix=np.loadtxt(file_path2,delimiter=",")
adj_matrix=np.loadtxt(file_path,delimiter=",")
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

# Calculate similarity scores
jaccard_scores = cr.jaccard_similarity(adj_matrix, node_index)
adamic_adar_scores = cr.adamic_adar_index(adj_matrix, node_index)

# Sort scores and get top 5 recommendations based on similarity
top_jaccard = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)[:5]
top_adamic_adar = sorted(adamic_adar_scores, key=lambda x: x[1], reverse=True)[:5]

print("Top 5 nodes based on Jaccard similarity:", top_jaccard)
print("Top 5 nodes based on Adamic/Adar index:", top_adamic_adar)

# init predict
recommended_nodes = cr.predict(model, adj_matrix, node_index, top_k=5, threshold=0.5)

# prinitng result
print(f"Recommended nodes for node {node_index}: {recommended_nodes}")
jaccard_score, adamic_adar_score = cr.aaj_accuracy(adj_matrix, node_index, recommended_nodes)
print(f"Average Jaccard Score: {jaccard_score}")
print(f"Average Adamic/Adar Score: {adamic_adar_score}")


vg.draw_graph_3d(adj_matrix,top_nodes=top_nodes,node_labels=node_labels,edge_weights=weight_matrix)
c=vg.bipartite_matrix_maker("graph_dataset.csv")
vg.show_bipartite_relationship(c)

# vg.draw_graph(adj_matrix,top_nodes=top_nodes,node_labels=node_labels,edge_weights=weight_matrix)
