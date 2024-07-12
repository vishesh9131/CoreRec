"""
models.py

This module defines the GraphTransformer class, a neural network model using Transformer architecture for processing graph data.

Classes:
    GraphTransformer: A neural network model for graph data using Transformer architecture.

Usage:
    from engine.models import GraphTransformer

    # Example usage
    model = GraphTransformer(num_layers=2, d_model=128, num_heads=4, d_feedforward=512, input_dim=10)
    output = model(torch.randn(1, 10))
"""

from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, ModuleList
import torch

# class GraphTransformer(Module):
#     def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True):
#         super(GraphTransformer, self).__init__()
#         self.num_weights = num_weights
#         self.use_weights = use_weights
#         self.input_linear = Linear(input_dim, d_model)
#         self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.output_linear = Linear(d_model, input_dim)
#         if self.use_weights:
#             self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])

#     def forward(self, x, weights=None):
#         x = x.float()
#         if self.use_weights:
#             if weights is not None:
#                 weighted_x = torch.zeros_like(x)
#                 for i, weight in enumerate(weights):
#                     weighted_x += self.weight_linears[i](x) * weight
#                 x = weighted_x
#             else:
#                 x = self.input_linear(x)
#         else:
#             x = self.input_linear(x)
#         x = self.transformer_encoder(x)
#         x = self.output_linear(x)
#         return x


"""
models.py

This module defines the NodeRecommender class, a neural network model for recommending nodes in a graph based on direct connections, neighborhood similarity, and graph structure.

Classes:
    NodeRecommender: A neural network model for recommending nodes in a graph.

Usage:
    from engine.models import NodeRecommender

    # Example usage
    model = NodeRecommender(input_dim=128, hidden_dim=64, output_dim=1)
    output = model(data)
"""

from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, ReLU, ModuleList
import torch
from torch_geometric.utils import degree

class GraphTransformerV2(Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, output_dim):
        super(GraphTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim

        # Define layers
        self.input_linear = Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = Linear(d_model + 1, output_dim)  # +1 for centrality score
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
