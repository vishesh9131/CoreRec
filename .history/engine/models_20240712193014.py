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