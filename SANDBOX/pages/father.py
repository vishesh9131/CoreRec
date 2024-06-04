# import streamlit as st
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# import core_rec as cs
# import vish_graphs as vg
# import matplotlib.pyplot as plt
# import os

# # Load data and model outside of the app function to avoid reloading
# @st.cache_data()
# def load_data_and_model():
#     adj_matrix = np.loadtxt('./SANDBOX/adj.csv', delimiter=",")
#     wgt_matrix = np.loadtxt('./SANDBOX/label.csv', delimiter=",")
#     # adj_matrix = np.loadtxt('adj.csv', delimiter=",")
#     # wgt_matrix = np.loadtxt('label.csv', delimiter=",")
#     df = pd.read_csv("./SANDBOX/labelele.csv")
#     col = df.values.flatten()
#     node_labels = {i: label for i, label in enumerate(col)}
#     model = cs.GraphTransformer(num_layers=2, d_model=128, num_heads=8, d_feedforward=512, input_dim=len(adj_matrix[0]), use_weights=True)
#     model.load_state_dict(torch.load('./SANDBOX/trained_model.pth'))
#     # model.load_state_dict(torch.load('./SANDBOX/megatron_e100.pth'))
#     # model.load_state_dict(torch.load('./SANDBOX/alpha_tuned_model_e1k_.pth'))
#     # model.load_state_dict(torch.load('./SANDBOX/GOAT_model_e10k_.pth'))
#     # model.load_state_dict(torch.load('./SANDBOX/natural_model_e10_.pth'))
#     # model.load_state_dict(torch.load('./SANDBOX/pushed_model_e50_.pth'))


#     # model.load_state_dict(torch.load('trained_model.pth'))

#     model.eval()
#     return adj_matrix, wgt_matrix, node_labels, model
# def app():
#     st.title('Test_A')
#     adj_matrix, wgt_matrix, node_labels, model = load_data_and_model()

#     label_options = list(node_labels.values())
#     selected_label = st.selectbox('Select node label:', label_options)
#     node_index = list(node_labels.keys())[list(node_labels.values()).index(selected_label)]

#     # Sliders for top_k and threshold
#     top_k = st.slider('Select top_k (How Many Peoples):', min_value=1, max_value=10, value=2)
#     threshold = st.slider('Select threshold (How Much Similarity):', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

#     if st.button('Run Model'):
#         with st.spinner('Running model...'):
#             recommended_indices = cs.predict(model, adj_matrix, node_index, top_k=top_k, threshold=threshold)
#             recommended_labels = [node_labels.get(idx, "Label not found") for idx in recommended_indices]
#             recommended_labels.reverse() 
#             st.write("Recommended nodes:")
#             for label in recommended_labels:
#                 st.success(label)

#             # Draw the graph with labels
#             fig = vg.draw_graph_3d(adj_matrix, node_labels=node_labels, top_nodes=[node_index], transparent_labeled=False, edge_weights=wgt_matrix)
#             st.pyplot(fig)

# app()

import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import core_rec as cs
import vish_graphs as vg
import matplotlib.pyplot as plt
import os

# Load data and model outside of the app function to avoid reloading
@st.cache_data()
def load_data_and_model(model_path):
    adj_matrix = np.loadtxt('./SANDBOX/adj.csv', delimiter=",")
    wgt_matrix = np.loadtxt('./SANDBOX/label.csv', delimiter=",")
    # adj_matrix = np.loadtxt('adj.csv', delimiter=",")
    # wgt_matrix = np.loadtxt('label.csv', delimiter=",")
    # df = pd.read_csv("labelele.csv")
    df = pd.read_csv("./SANDBOX/labelele.csv")

    col = df.values.flatten()
    node_labels = {i: label for i, label in enumerate(col)}
    model = cs.GraphTransformer(num_layers=2, d_model=128, num_heads=8, d_feedforward=512, input_dim=len(adj_matrix[0]), use_weights=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return adj_matrix, wgt_matrix, node_labels, model

def app():
    st.title('Test_A')

    # Model selection
    model_options = {
        "Trained Model": './SANDBOX/trained_model.pth',
        "Megatron Model_e100": './SANDBOX/megatron_e100.pth',
        "Alpha Tuned Model_e1k": './SANDBOX/alpha_tuned_model_e1k_.pth',
        "GOAT Model_e10k": './SANDBOX/GOAT_model_e10k_.pth',
        "Natural Model_e10": './SANDBOX/natural_model_e10_.pth',
        "Pushed Model_e50": './SANDBOX/pushed_model_e50_.pth',
        "Massive Surya_e10k": './SANDBOX/massive_surya_e10k.pth'

        # "Trained Model": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/trained_model.pth',
        # "Megatron Model_e100": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/megatron_e100.pth',
        # "Alpha Tuned Model_e1k": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/alpha_tuned_model_e1k_.pth',
        # "GOAT Model_e10k": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/GOAT_model_e1k_.pth',
        # "Natural Model_e10": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/natural_model_e10_.pth',
        # "Pushed Model_e50": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/pushed_model_e50_.pth',
        # "Massive Surya_e10k": '/Users/visheshyadav/Documents/GitHub/CoreRec/SANDBOX/massive_surya_e10k.pth'
    }
    selected_model = st.selectbox('Select model:', list(model_options.keys()))
    model_path = model_options[selected_model]

    adj_matrix, wgt_matrix, node_labels, model = load_data_and_model(model_path)

    label_options = list(node_labels.values())
    selected_label = st.selectbox('Select node label:', label_options)
    node_index = list(node_labels.keys())[list(node_labels.values()).index(selected_label)]

    # Sliders for top_k and threshold
    top_k = st.slider('Select top_k (How Many Peoples):', min_value=1, max_value=10, value=2)
    threshold = st.slider('Select threshold (How Much Similarity):', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if st.button('Run Model'):
        with st.spinner('Running model...'):
            recommended_indices = cs.predict(model, adj_matrix, node_index, top_k=top_k, threshold=threshold)
            recommended_labels = [node_labels.get(idx, "Label not found") for idx in recommended_indices]
            # recommended_labels.reverse() 
            st.write("Recommended nodes:")
            for label in recommended_labels:
                st.success(label)

            # Draw the graph with labels
            fig = vg.draw_graph(adj_matrix, node_labels=node_labels, top_nodes=[node_index], transparent_labeled=False,recommended_nodes=recommended_indices)
            st.pyplot(fig)

app()