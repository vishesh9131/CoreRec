# This is a time capsule module for the corerec 
# ###############################################################################################################
#                           --CoreRec: Connecting to the Unseen--                            
# CoreRec module is designed for graph-based recommendation systems using neural network architectures. It includes:
#     1. GraphTransformer: A neural network model using Transformer architecture for processing graph data.
#     2. GraphDataset: A custom dataset class for handling graph data.
#     3. train_model: A function to train models with options for custom loss functions and training procedures.
#     4. predict: Functions to predict top-k nodes based on model outputs, with optional thresholding.
#     5. draw_graph: A function to visualize graphs with options to highlight top nodes and recommended nodes.
# Note: This module integrates PyTorch for model training and evaluation, and NetworkX for graph manipulation.
# ###############################################################################################################

from common_import import *
from async_ddp import *
from torch_geometric.data import Data

class GraphTransformer(Module):
    '''
    This is a transformer model is used from ;
    link :  https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py 
    '''
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True):
        super(GraphTransformer, self).__init__()
        self.num_weights = num_weights
        self.use_weights = use_weights
        self.input_linear = Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = Linear(d_model, input_dim)
        if self.use_weights:
            self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])

    def forward(self, x, weights=None):
        x = x.float()
        if self.use_weights:
            if weights is not None:
                weighted_x = torch.zeros_like(x)
                for i, weight in enumerate(weights):
                    weighted_x += self.weight_linears[i](x) * weight
                x = weighted_x
            else:
                x = self.input_linear(x)
        else:
            x = self.input_linear(x)  # Ensure input is transformed to d_model dimension
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x

# Custom Dataset for Graph Data
class GraphDataset(Dataset):
    def __init__(self, adj_matrix, weight_matrix=None):
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)  # Ensure float32
        if weight_matrix is not None:
            self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)  # Ensure float32
        else:
            self.weight_matrix = None

    def __len__(self):
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        node_features = self.adj_matrix[idx]
        edge_index = torch.nonzero(self.adj_matrix).t().contiguous()
        data = Data(x=node_features, edge_index=edge_index)
        if self.weight_matrix is not None:
            weights = self.weight_matrix[idx]
            data.weights = weights
        return data

# Training Loop
def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs = inputs.float()  # Ensure inputs are float
            targets = targets.float()  # Ensure targets are float
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


# THIS TRAIN FN IS A RESCUE BRANCH TO ABOVE FN DONOT DELETE IT

# def train_model(model, data_loader, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in data_loader:
#             inputs, targets = batch
#             inputs = inputs.float()
#             targets = targets.float()

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = custom_loss_fn(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         average_loss = total_loss / len(data_loader)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")


def predict(model, graph, node_index,top_k=5):
    
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0)  # Get the input node's features
        output = model(input_data)
        scores = output.squeeze().numpy()
    # Get top-k node indices based on scores
    recommended_indices = scores.argsort()[-top_k:][::-1]
    return recommended_indices

def predict(model, graph, node_index, top_k=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0)  # Get the input node's features
        output = model(input_data)
        scores = output.squeeze().numpy()
    
    # Apply threshold
    recommended_indices = [i for i, score in enumerate(scores) if score > threshold]
    recommended_indices = sorted(recommended_indices, key=lambda i: scores[i], reverse=True)[:top_k]
    return recommended_indices






# Graph Drawing Function [its in defaulter mode]
def draw_graph(adj_matrix, top_nodes, recommended_nodes=None):
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if recommended_nodes is not None and node in recommended_nodes:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12)

    # Highlight top nodes with a different shape
    if top_nodes is not None:
        top_node_color = 'green'
        nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_color=top_node_color, node_size=500, node_shape='s')

    plt.title("Recommended Nodes Highlighted in Blue and Top Nodes in Red")
    plt.show()
    



def jaccard_similarity(graph, node):
    G = nx.from_numpy_array(graph)  # Use from_numpy_array if from_numpy_matrix gives issues
    scores = []
    neighbors = set(G.neighbors(node))
    for n in G.nodes():
        if n != node:
            neighbors_n = set(G.neighbors(n))
            intersection = len(neighbors & neighbors_n)
            union = len(neighbors | neighbors_n)
            score = intersection / union if union != 0 else 0
            scores.append((n, score))
    return scores

def adamic_adar_index(graph, node):
    G = nx.from_numpy_array(graph)
    scores = []
    neighbors = set(G.neighbors(node))
    for n in G.nodes():
        if n != node:
            neighbors_n = set(G.neighbors(n))
            shared_neighbors = neighbors & neighbors_n
            score = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in shared_neighbors if len(list(G.neighbors(nn))) > 1)
            scores.append((n, score))
    return scores



def aaj_accuracy(graph, node_index, recommended_indices):
    G = nx.from_numpy_array(graph)
    jaccard_scores = []
    adamic_adar_scores = []

    # Calculate Jaccard and Adamic/Adar for recommended nodes
    for rec_node in recommended_indices:
        # Jaccard
        preds = list(nx.jaccard_coefficient(G, [(node_index, rec_node)]))
        if preds:
            jaccard_scores.append(preds[0][2])  # preds[0][2] is the Jaccard coefficient

        # Adamic/Adar
        preds = list(nx.adamic_adar_index(G, [(node_index, rec_node)]))
        if preds:
            adamic_adar_scores.append(preds[0][2])  # preds[0][2] is the Adamic/Adar index

    # Calculate average scores
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    avg_adamic_adar = np.mean(adamic_adar_scores) if adamic_adar_scores else 0

    return avg_jaccard, avg_adamic_adar



# XAI
def explainable_predict(model, graph, node_index, top_k=5, threshold=0.5):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(graph[node_index]).unsqueeze(0)
        output = model(input_data)
        scores = output.squeeze().numpy()

    # Apply threshold and get top-k recommendations
    recommended_indices = [i for i, score in enumerate(scores) if score > threshold]
    recommended_indices = sorted(recommended_indices, key=lambda i: scores[i], reverse=True)[:top_k]

    explanations = []
    G = nx.from_numpy_array(graph)
    user_neighbors = set(G.neighbors(node_index))

    for idx in recommended_indices:
        node_neighbors = set(G.neighbors(idx))
        jaccard = len(user_neighbors & node_neighbors) / len(user_neighbors | node_neighbors) if len(user_neighbors | node_neighbors) > 0 else 0
        adamic_adar = sum(1 / np.log(len(list(G.neighbors(nn)))) for nn in user_neighbors & node_neighbors if len(list(G.neighbors(nn))) > 1)

        # Generate natural language explanations
        explanation_text = f"The recommendation is based on the similarity of node {idx} to your interests and its connections to relevant nodes."

        explanation = {
            "node": idx,
            "score": scores[idx],
            "jaccard_similarity": jaccard,
            "adamic_adar_index": adamic_adar,
            "explanation": explanation_text
        }
        explanations.append(explanation)

    return recommended_indices, explanations
