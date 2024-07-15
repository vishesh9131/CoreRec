from common_import import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Check for available device (CPU in this case)
device = torch.device('cpu')

class GraphTransformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_feedforward: int, input_dim: int, use_weights: bool = False):
        super(GraphTransformer, self).__init__()
        self.use_weights = use_weights
        self.input_linear = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        x = x.float()
        if self.use_weights and weights is not None:
            x = x * weights.float()
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x

class GraphDataset(Dataset):
    def __init__(self, adj_matrix: np.ndarray, weight_matrix: np.ndarray = None):
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32) if weight_matrix is not None else None

    def __len__(self) -> int:
        return len(self.adj_matrix)

    def __getitem__(self, idx: int):
        node_features = self.adj_matrix[idx]
        if self.weight_matrix is not None:
            return node_features, self.weight_matrix[idx]
        return node_features, node_features

def train_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader):.4f}")

def predict(model: nn.Module, graph: np.ndarray, node_index: int, top_k: int = 5, threshold: float = 0.5) -> list:
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0).to(device)
        scores = model(input_data).squeeze().numpy()
    recommended_indices = np.argwhere(scores > threshold).flatten()
    return recommended_indices[np.argsort(scores[recommended_indices])[-top_k:][::-1]]

def draw_graph(adj_matrix: np.ndarray, top_nodes: list, recommended_nodes: list = None):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G)
    node_colors = ['red' if node in recommended_nodes else 'skyblue' for node in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, alpha=0.8)
    if top_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color='green', node_size=500, node_shape='s')
    plt.title("Recommended Nodes Highlighted in Blue and Top Nodes in Red")
    plt.show()

def similarity_scores(graph: np.ndarray, node: int, metric: str) -> list:
    G = nx.from_numpy_array(graph)
    neighbors = set(G.neighbors(node))
    scores = []
    for n in G.nodes():
        if n == node:
            continue
        neighbors_n = set(G.neighbors(n))
        if metric == 'jaccard':
            union = neighbors | neighbors_n
            score = len(neighbors & neighbors_n) / len(union) if union else 0
        elif metric == 'adamic_adar':
            score = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in neighbors & neighbors_n if len(list(G.neighbors(nn))) > 1)
        scores.append((n, score))
    return scores

def aaj_accuracy(graph: np.ndarray, node_index: int, recommended_indices: list) -> tuple:
    G = nx.from_numpy_array(graph)
    jaccard_scores = [list(nx.jaccard_coefficient(G, [(node_index, rec_node)]))[0][2] for rec_node in recommended_indices]
    adamic_adar_scores = [list(nx.adamic_adar_index(G, [(node_index, rec_node)]))[0][2] for rec_node in recommended_indices]
    return np.mean(jaccard_scores) if jaccard_scores else 0, np.mean(adamic_adar_scores) if adamic_adar_scores else 0

def explainable_predict(model: nn.Module, graph: np.ndarray, node_index: int, top_k: int = 5, threshold: float = 0.5) -> tuple:
    recommended_indices = predict(model, graph, node_index, top_k, threshold)
    explanations = []
    G = nx.from_numpy_array(graph)
    user_neighbors = set(G.neighbors(node_index))
    for idx in recommended_indices:
        node_neighbors = set(G.neighbors(idx))
        jaccard = len(user_neighbors & node_neighbors) / len(user_neighbors | node_neighbors) if len(user_neighbors | node_neighbors) > 0 else 0
        adamic_adar = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in user_neighbors & node_neighbors if len(list(G.neighbors(nn))) > 1)
        explanations.append({
            "node": idx,
            "score": graph[node_index][idx],
            "jaccard_similarity": jaccard,
            "adamic_adar_index": adamic_adar,
            "explanation": f"The recommendation is based on the similarity of node {idx} to your interests and its connections to relevant nodes."
        })
    return recommended_indices, explanations

def format_predictions(predictions):
    formatted_results = []
    for pred in predictions:
        formatted_results.append(
            f"Node: {pred['node']}\n"
            f"Score: {pred['score']:.4f}\n"
            f"Jaccard Similarity: {pred['jaccard_similarity']:.4f}\n"
            f"Adamic/Adar Index: {pred['adamic_adar_index']:.4f}\n"
            f"Explanation: {pred['explanation']}\n"
            "-----------------------------"
        )
    return "\n".join(formatted_results)

if __name__ == '__main__':
    adj_matrix = np.random.rand(100, 100)  # Example larger adjacency matrix
    weight_matrix = np.random.rand(100, 100)  # Example larger weight matrix

    dataset = GraphDataset(adj_matrix, weight_matrix)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)  # Use more workers for larger data

    model = GraphTransformer(num_layers=4, d_model=128, num_heads=8, d_feedforward=256, input_dim=100, use_weights=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, data_loader, criterion, optimizer, num_epochs=20)

    # Predict and explain
    node_index = 0
    recommended_indices, explanations = explainable_predict(model, adj_matrix, node_index)
    print("Recommended Indices:", recommended_indices)
    print("Explanations:")
    print(format_predictions(explanations))

    # Example of parallel processing for similarity scores
    def compute_similarity_scores(node):
        return similarity_scores(adj_matrix, node, 'jaccard')

    nodes_to_compute = list(range(100))  # Example nodes
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compute_similarity_scores, nodes_to_compute))

    print("Similarity Scores for Nodes:", results)