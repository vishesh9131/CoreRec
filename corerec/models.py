"""
models.py

This module defines the GraphTransformer, GraphSAGE, GAT, GNN, HAN, and GCF classes, neural network models using various architectures for processing graph data.

Classes:
    GraphTransformer: A neural network model for graph data using Transformer architecture.
    GraphSAGE: A neural network model for graph data using GraphSAGE architecture.
    GAT: A neural network model for graph data using Graph Attention Network architecture.
    GNN: A general neural network model for graph data.
    HAN: A neural network model for heterogeneous graph data using Heterogeneous Graph Attention Network architecture.
    GCF: A neural network model for collaborative filtering using Graph Convolutional Factorization.

Usage:
    from engine.models import GraphTransformer, GraphSAGE, GAT, GNN, HAN, GCF

    # Example usage for GraphTransformer
    model = GraphTransformer(num_layers=2, d_model=128, num_heads=4, d_feedforward=512, input_dim=10)
    output = model(torch.randn(1, 10))

    # Example usage for GraphSAGE
    model = GraphSAGE(in_channels=10, out_channels=128, num_layers=2, aggr='mean')
    output = model(data)

    # Example usage for GAT
    model = GAT(in_channels=10, out_channels=128, num_layers=2, heads=4)
    output = model(data)

    # Example usage for GNN
    model = GNN(in_channels=10, hidden_channels=64, out_channels=128, num_layers=2)
    output = model(data)

    # Example usage for HAN
    model = HAN(in_channels=10, out_channels=128, num_heads=8, num_metapaths=3)
    output = model(data)

    # Example usage for GCF
    model = GCF(num_users=1000, num_items=500, embedding_dim=64, num_layers=2)
    output = model(user_indices, item_indices)
"""
from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, ReLU, ModuleList, Embedding
from torch_geometric.utils import degree
# from cr_pkg import sage_conv, gat_conv, gcn_conv, han_conv
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HANConv
import torch

class GraphTransformer(Module):
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
            x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x)
        return x

class GraphSAGE(Module):
    def __init__(self, in_channels, out_channels, num_layers, aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels, aggr=aggr))
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return x

class GAT(Module):
    def __init__(self, in_channels, out_channels, num_layers, heads=1):
        super(GAT, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, out_channels // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(out_channels, out_channels // heads, heads=heads))
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return x
    
class GNN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return x

class HAN(Module):
    def __init__(self, in_channels, out_channels, num_heads, num_metapaths):
        super(HAN, self).__init__()
        self.convs = ModuleList()
        for _ in range(num_metapaths):
            self.convs.append(HANConv(in_channels, out_channels, heads=num_heads))
        self.lin = Linear(out_channels * num_heads, out_channels)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index_dict = data.x, data.edge_index_dict
        out = []
        for conv, edge_index in zip(self.convs, edge_index_dict.values()):
            out.append(conv(x, edge_index))
        x = torch.cat(out, dim=1)
        x = self.lin(x)
        x = self.relu(x)
        return x

class GCF(Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(GCF, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_dim)
        self.item_embedding = Embedding(num_items, embedding_dim)
        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(embedding_dim, embedding_dim))
        self.relu = ReLU()

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        x = torch.cat([user_emb, item_emb], dim=0)
        edge_index = self.build_edge_index(user_indices, item_indices, x.size(0))
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        user_emb, item_emb = torch.split(x, [user_emb.size(0), item_emb.size(0)], dim=0)
        return user_emb, item_emb

    def build_edge_index(self, user_indices, item_indices, num_nodes):
        user_indices = user_indices + num_nodes // 2
        edge_index = torch.stack([torch.cat([user_indices, item_indices]), torch.cat([item_indices, user_indices])], dim=0)
        return edge_index
    










######################-Work-in-Progress-##################### 
class GraphTransformerV2_test(Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim):
        super(GraphTransformerV2, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # self.output_dim = output_dim

        # Define layers
        self.input_linear = Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.output_linear = Linear(d_model + 1, output_dim)  # +1 for centrality score
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Direct Connections: Aggregate features from direct neighbors
        row, col = edge_index
        agg_features = torch.zeros_like(x)
        agg_features.index_add_(0, row, x[col])

        # Apply input linear layer and activation
        x = self.input_linear(agg_features)
        x = self.relu(x)

        # Neighborhood Similarity: Aggregate features from neighbors of neighbors
        agg_features = torch.zeros_like(x)
        agg_features.index_add_(0, row, x[col])

        # Apply transformer encoder
        x = self.transformer_encoder(agg_features)

        # Graph Structure: Centrality score
        centrality = degree(edge_index[0], x.size(0), dtype=x.dtype).view(-1, 1)

        # Concatenate centrality score with node features
        x = torch.cat([x, centrality], dim=1)

        # Final linear layer
        x = self.output_linear(x)

        return x

