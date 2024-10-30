import csv
import numpy as np
import pandas as pd


# Read the CSV file
file_path = 'src/SANDBOX/recsys/test5.csv'
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    movies = list(reader)

# Extract movie details
movie_details = []
for movie in movies:
    actors = set(movie['actors'].split(' | '))
    genres = set(movie['genre'].split(' | '))
    movie_details.append((movie['title'], actors, genres))

# Initialize the adjacency matrix
num_movies = len(movie_details)
adj_matrix = np.zeros((num_movies, num_movies), dtype=int)

# Populate the adjacency matrix
for i in range(num_movies):
    for j in range(i + 1, num_movies):
        shared_actors = movie_details[i][1].intersection(movie_details[j][1])
        shared_genres = movie_details[i][2].intersection(movie_details[j][2])
        
        # Assign weights: 2 for shared actors, 1 for shared genres
        weight = 0
        if shared_actors:
            weight += 2
        if shared_genres:
            weight += 1
        
        adj_matrix[i, j] = weight
        adj_matrix[j, i] = weight

# # Save the adjacency matrix to a CSV file without movie names
# output_file_path = 'src/SANDBOX/recsys/test5.csv'
# with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in adj_matrix:
#         writer.writerow(row)

# print(f"Adjacency matrix saved to {output_file_path} without movie names")

df = pd.read_csv(file_path)
label = df['title']

import corerec.vish_graphs as vg

top_nodes = vg.find_top_nodes(adj_matrix,num_nodes=3)
vg.draw_graph_3d(adj_matrix,
                 transparent_labeled=False,
                 node_labels=label,
                 top_nodes=top_nodes)

from corerec.Tmodel import GraphTransformerV2
model = GraphTransformerV2(num_layers=3,d_model=128,num_heads=8,d_feedforward=512,input_dim=128,num_weights=10,use_weights=True,dropout=0.1)

import torch

predict = model(adj_matrix,torch.eye(len(adj_matrix)),None)
print(predict)