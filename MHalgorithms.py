'''
                        -Metropolis-Hastings Algorithm-
The Metropolis-Hastings community detection algorithm iteratively adjusts 
community assignments based on the structure of the graph and a probabilistic model. 
Nodes are more likely to join communities where they have more connections, leading to 
a clustering effect where densely connected nodes are grouped together. This example 
simplifies the process, but in practice, the algorithm would run for many iterations to 
stabilize the community assignments.
'''
import numpy as np
import networkx as nx
import random
from collections import defaultdict
import streamlit as st
import matplotlib.pyplot as plt
import vish_graphs as vg
# # from scipy.spatial import ConvexHull
import plotly.graph_objects as go

# Generate random graphs and load adjacency matrices
file_path1 = vg.generate_random_graph(30, file_path="graph_dataset.csv",seed=23)
user_user_matrix = np.loadtxt(file_path1, delimiter=",")

file_path2 = vg.generate_random_graph(30, file_path="graph_dataset2.csv",seed=23)
user_content_matrix = np.loadtxt(file_path2, delimiter=",")

def generate_random_names(num_names):
    names = [
        "Kaa", "Kha", "Ga", "Gha", "Cha",
        "Chha", "Ja", "Jha", "Ta", "Tha",
        "Da", "Dha", "Na", "Pa", "Pha",
        "Ba", "Bha", "Ma", "Ya", "Ra",
        "La", "Va", "Sha", "Sa"
    ]
    return random.sample(names, num_names)

def metropolis_hastings_community_detection(graph, num_communities, iterations):
    community_names = generate_random_names(num_communities)
    communities = {node: random.randint(0, num_communities - 1) for node in graph.nodes()}
    
    for _ in range(iterations):
        for node in graph.nodes():
            current_community = communities[node]
            new_community = random.randint(0, num_communities - 1)
            
            if new_community != current_community:
                current_neighbors = [n for n in graph.neighbors(node) if communities[n] == current_community]
                new_neighbors = [n for n in graph.neighbors(node) if communities[n] == new_community]
                acceptance_prob = min(1, (len(new_neighbors) + 1) / (len(current_neighbors) + 1))
                
                if random.random() < acceptance_prob:
                    communities[node] = new_community
    
    named_communities = {node: community_names[community] for node, community in communities.items()}
    
    return named_communities

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(user_user_matrix)

# Run the Metropolis-Hastings community detection
communities = metropolis_hastings_community_detection(G, num_communities=3, iterations=1000)

def generate_representations(communities, num_communities):
    community_names = list(set(communities.values()))
    num_communities = len(community_names)
    
    user_representations = defaultdict(lambda: np.zeros(num_communities))
    content_representations = defaultdict(lambda: np.zeros(num_communities))
    
    community_index = {name: idx for idx, name in enumerate(community_names)}
    
    for user, community_name in communities.items():
        community_idx = community_index[community_name]
        user_representations[user][community_idx] = 1
    
    for user in range(user_content_matrix.shape[0]):
        for content in range(user_content_matrix.shape[1]):
            if user_content_matrix[user, content] == 1:
                content_representations[content] += user_representations[user]
    
    return user_representations, content_representations

user_reps, content_reps = generate_representations(communities, num_communities=3)

def recommend_content(user_id, user_reps, content_reps, top_k=3):
    user_vector = user_reps[user_id]
    content_scores = {content: np.dot(user_vector, content_vector) for content, content_vector in content_reps.items()}
    recommended_content = sorted(content_scores, key=content_scores.get, reverse=True)[:top_k]
    return recommended_content


# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(user_user_matrix)

# Run the Metropolis-Hastings community detection
communities = metropolis_hastings_community_detection(G, num_communities=3, iterations=1000)

# Generate user and content representations
user_reps, content_reps = generate_representations(communities, num_communities=3)

# Streamlit UI enhancements
st.title("SimClusters: Community-Based Representations for Recommendations")

col1, col2 = st.columns(2)
with col1:
    st.header("User-User Adjacency Matrix")
    st.write(user_user_matrix)

with col2:
    st.header("User-Content Interaction Matrix")
    st.write(user_content_matrix)


# user_id = st.number_input("Enter User ID for Recommendations", min_value=0, max_value=len(user_user_matrix)-1, step=1)
user_id = st.number_input("Enter User ID for Recommendations", min_value=0, max_value=len(user_user_matrix)-1, step=1, key="user_id_input")
if st.button("Recommend Content"):
    recommended_content = recommend_content(user_id, user_reps, content_reps)
    st.write(f"Recommended content for user {user_id}: {recommended_content}")


# Using Plotly for interactive graphs
def plotly_user_content_graph(user_content_matrix):
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])  # Placeholder for actual data
    return fig

st.header("User-Content Interaction Graph")
fig_user_content = plotly_user_content_graph(user_content_matrix)
st.plotly_chart(fig_user_content, use_container_width=True)

def plotly_community_graph(G, communities):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    text = []
    community_list = list(set(communities.values()))  # Unique list of community names
    community_to_index = {name: idx for idx, name in enumerate(community_list)}  # Map community names to indices

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"{communities[node]}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[community_to_index[communities[node]] for node in G.nodes()],  # Use community indices as colors
            colorbar=dict(thickness=15)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

st.header("Detected Communities")
fig_community = plotly_community_graph(G, communities)
st.plotly_chart(fig_community, use_container_width=True)

# Credit
with st.expander("menu"):
    st.write("Powered by vishGraphs")

if st.button('Show more'):
    st.write("More information here.")


tab1, tab2, tab3 = st.tabs(["Home", "Settings", "About"])
with tab1:
    st.write("Welcome to the Home tab.")
with tab2:
    st.write("Adjust settings here.")
with tab3:
    st.write("Learn more about this app.")


if st.button('Show Alert'):
    st.error('This is an alert!')



st.write("---")  # Draw a line
col1, col2 = st.columns([1, 3])
with col1:
    st.write("Powered by Streamlit")
with col2:
    st.write("More info at [GitHub](https://github.com)")
