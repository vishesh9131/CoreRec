import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import core_rec as cs
import vish_graphs as vg
import random
import matplotlib.pyplot as plt
import os



def app():
    st.title('CoreRec')

    # Load the CSV file into a DataFrame
    adj_matrix = np.loadtxt('SANDBOX/adj.csv', delimiter=",")

    wgt_matrix = np.loadtxt('SANDBOX/label.csv', delimiter=",")

    df = pd.read_csv("SANDBOX/labelele.csv")
    # df = pd.read_csv("labelele.csv")
    # df = pd.read_csv("labelele.csv")
    col = df.values.flatten()
    node_labels = {i: label for i, label in enumerate(col)}

    # Load the pre-trained model
    model = cs.GraphTransformer(num_layers=2, d_model=128, num_heads=8, d_feedforward=512, input_dim=len(adj_matrix[0]), use_weights=True)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Load node labels
    df = pd.read_csv("labelele.csv")
    col = df.values.flatten()
    node_labels = {i: label for i, label in enumerate(col)}

    # Find top nodes
    top_nodes = vg.find_top_nodes(adj_matrix, 4)

    # # ML
    # # Convert adjacency matrix to dataset
    graph_dataset = cs.GraphDataset(adj_matrix)
    data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

    # # Define model parameters
    # num_layers = 2
    # d_model = 128
    # num_heads = 8
    # d_feedforward = 512
    # input_dim = len(adj_matrix[0])

    # # Initialize model, loss function, and optimizer
    # model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim, use_weights=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    label_options = list(node_labels.values())
    selected_label = st.selectbox('Select node label:', label_options)
    node_index = list(node_labels.keys())[list(node_labels.values()).index(selected_label)]
    num_epochs = 200
    num_iterations = 12

    # Button to trigger computation
    if st.button('Run Model'):
        with st.spinner('Running model...'):
            progress_bar = st.progress(0)
            status_text = st.empty()  # Create an empty placeholder for status text
            all_recommended_labels = []
            for i in range(num_iterations):
                cs.train_model(model, data_loader, criterion, optimizer, num_epochs)
                recommended_nodes = cs.predict(model, adj_matrix, node_index, top_k=2, threshold=1.0)
                recommended_labels = [node_labels[node] for node in recommended_nodes]
                all_recommended_labels.extend(recommended_labels)
                progress = (i + 1) / num_iterations
                progress_bar.progress(progress)
                # Define messages based on progress thresholds
                if progress < 0.25:
                    message = "Training in progress...."
                elif progress <0.35:
                    message = "Optimizing parameters..."
                elif progress < 0.50:
                    message = "Finding top nodes..."
                elif progress < 0.75:
                    message = "Almost there! Fine-tuning the model"
                elif progress < 0.85:
                    message = "Recommendation in progress...."
                else:
                    message = "Recommendation complete!"

                status_text.text(f"{message} {int(progress * 100)}%")   
                # status_text.text(f"{random_texts[i % len(random_texts)]} {int(progress * 100)}%")
                # status_text.text(f"{len.choice(random_texts)} {int(progress * 100)}%")  # Update status text with random text and percentage
            progress_bar.empty()
            status_text.empty()  # Clear the status text after completion

            # Count the most frequent recommended labels
            label_counts = Counter(all_recommended_labels)
            most_common_labels = label_counts.most_common()
            st.write("Most frequently recommended nodes:")
            for label, count in most_common_labels:
                st.write(f"{label}: {count} times")

            # Draw a bar graph of the frequency of recommended labels
            labels, counts = zip(*most_common_labels)
            fig, ax = plt.subplots()
            ax.bar(labels, counts)
            ax.set_xlabel('Labels')
            ax.set_ylabel('Frequency')
            ax.set_title('Frequency of Recommended Labels')
            st.pyplot(fig)

            # Draw the graph with labels
            fig = vg.draw_graph_3d(adj_matrix, node_labels=node_labels, top_nodes=top_nodes, transparent_labeled=False, edge_weights=wgt_matrix)
            st.pyplot(fig)