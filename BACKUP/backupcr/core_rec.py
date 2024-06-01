
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

class GraphTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim):
        super(GraphTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)  # Linear layer to map input_dim to d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, input_dim)  # Output layer for node scores

    def forward(self, x):
        x = x.float()  
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x

# Custom Dataset for Graph Data
class GraphDataset(Dataset):
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        # Here, let's assume we're trying to predict connections for each node.
        node_features = self.adj_matrix[idx]
        targets = self.adj_matrix[idx]  # Use adjacency matrix row as the target.
        return node_features, targets

# Training Loop
def train_model(model, data_loader, criterion=False, optimizer=False, num_epochs=False):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, targets = batch
            inputs = inputs.float()
            targets = targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


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


# Graph Drawing Function
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
