import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
import networkx as nx
import numpy as np
from corerec.datasets import GraphDataset
from corerec.Tmodel import GraphTransformerV2
from corerec.cr_boosters.adam import Adam
import corerec.core_rec as cr
import corerec.vish_graphs as vg

# Streamlit app title
st.title("Video Recommendation System")

# Step 1: Load and preprocess the CA Videos dataset
st.header("Step 1: Load Dataset")
try:
    ca_videos_data = pd.read_csv('nanodata.csv')  # Load your dataset
except pd.errors.ParserError as e:
    print(f"Error loading data: {e}")

# Step 2: Create edges based on videos from the same channel
st.header("Step 2: Preprocess Data")
edges = []
channels = ca_videos_data['channel_title'].unique()
for channel in channels:
    channel_videos = ca_videos_data[ca_videos_data['channel_title'] == channel]
    video_ids = channel_videos['video_id'].values
    for i in range(len(video_ids)):
        for j in range(i + 1, len(video_ids)):
            edges.append((video_ids[i], video_ids[j]))

G = nx.Graph()
G.add_edges_from(edges)
adj_matrix = nx.to_numpy_array(G)
weight_matrix = np.ones_like(adj_matrix)  # Example weight matrix with all ones

graph_dataset = GraphDataset(adj_matrix, weight_matrix)
data_loader = DataLoader(graph_dataset, batch_size=32, shuffle=True)
st.write("Data preprocessed successfully!")

# Step 3: Initialize the model
st.header("Step 3: Initialize Model")
model = GraphTransformerV2(num_layers=2, d_model=128, num_heads=4, d_feedforward=512, input_dim=adj_matrix.shape[1])
st.write("Model initialized successfully!")

# Step 4: Train the model
st.header("Step 4: Train Model")
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
num_epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10, step=1)

if st.button("Train Model"):
    cr.train_model(model, data_loader, criterion, optimizer, num_epochs=num_epochs)
    torch.save(model.state_dict(), "trained_model.pth")
    st.write("Model trained and saved successfully!")

# Step 5: Make predictions
st.header("Step 5: Make Predictions")
node_index = st.number_input("Node index for recommendations", min_value=0, max_value=adj_matrix.shape[0]-1, value=0, step=1)
top_k = st.number_input("Top K recommendations", min_value=1, max_value=20, value=5, step=1)
threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Generate Recommendations"):
    model.load_state_dict(torch.load("trained_model.pth"))
    recommended_nodes = cr.predict(model, adj_matrix, node_index, top_k=top_k, threshold=threshold)
    st.write(f"Recommended nodes for node {node_index}: {recommended_nodes}")

    # Like/Dislike feature
    for node in recommended_nodes:
        st.write(f"Recommendation: {node}")
        if st.button(f"Like {node}"):
            st.write(f"You liked {node}")
        if st.button(f"Dislike {node}"):
            st.write(f"You disliked {node}")

# # Step 6: Draw the graph
# st.header("Step 6: Visualize Graph")
# if st.button("Draw Graph"):
#     vg.draw_graph_3d(adj_matrix, recommended_nodes=recommended_nodes, transparent_labeled=False, edge_weights=weight_matrix)
#     st.write("Graph drawn successfully!")