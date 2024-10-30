import torch
from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder, ModuleList, Dropout, LayerNorm
import torch.nn.functional as F

class GraphTransformerV2(Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True, dropout=0.1):
        super(GraphTransformerV2, self).__init__()
        self.num_weights = num_weights
        self.use_weights = use_weights
        self.input_linear = Linear(input_dim, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_linear = Linear(d_model, input_dim)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        if self.use_weights:
            self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])

    def forward(self, x, adjacency_matrix, graph_metrics, weights=None):
        # Ensure adjacency_matrix is a FloatTensor
        adjacency_matrix = adjacency_matrix.float()

        # Check dimensions before matrix multiplication
        if adjacency_matrix.size(1) != x.size(1):
            raise ValueError(f"Dimension mismatch: adjacency_matrix has {adjacency_matrix.size(1)} nodes, but x has {x.size(1)} nodes.")

        try:
            # Direct Connections
            direct_scores = adjacency_matrix @ x  # Matrix multiplication to get direct connection scores

            # Neighborhood Similarity
            neighborhood_similarity = self.compute_neighborhood_similarity(adjacency_matrix, x)

            # Graph Structure
            graph_structure_scores = graph_metrics @ x  # Use precomputed graph metrics

            # Combine DNG scores
            dng_scores = direct_scores + neighborhood_similarity + graph_structure_scores

            # Optional weighted processing
            if self.use_weights and weights is not None:
                weighted_x = torch.zeros_like(x)
                for i, weight in enumerate(weights.T):
                    weighted_x += self.weight_linears[i](x) * weight.unsqueeze(1)
                x = weighted_x
            else:
                x = self.input_linear(x)

            x = self.layer_norm(x)
            x = self.transformer_encoder(x)
            x = self.output_linear(x)
            x = self.dropout(x)

            # Combine with DNG scores
            final_scores = F.relu(x + dng_scores)
            return final_scores

        except RuntimeError as e:
            print(f"RuntimeError during forward pass: {e}")
            print(f"x shape: {x.shape}, adjacency_matrix shape: {adjacency_matrix.shape}, graph_metrics shape: {graph_metrics.shape}")
            raise

    def compute_neighborhood_similarity(self, adjacency_matrix, x):
        # Jaccard similarity (simplified)
        intersection = adjacency_matrix @ adjacency_matrix
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        col_sums = adjacency_matrix.sum(dim=0, keepdim=True)
        union = row_sums + col_sums - intersection
        similarity = intersection / (union + 1e-6)  # Add small value to avoid division by zero
        return similarity @ x



# # Example usage (Do not consider it in production -vishesh)
# input_dim = 4 
# d_model = 8
# num_layers = 2
# num_heads = 2
# d_feedforward = 16
# x = torch.tensor([[0.1, 0.2, 0.3, 0.4],
#                   [0.5, 0.6, 0.7, 0.8]])
# adjacency_matrix = torch.tensor([[1, 0],
#                                  [0, 1]])  # Simplified adjacency matrix
# graph_metrics = torch.tensor([[0.5, 0.5],
#                               [0.5, 0.5]])  # Simplified graph metrics
# 
# model = GraphTransformerV2(num_layers=num_layers, 
#                             d_model=d_model, 
#                             num_heads=num_heads, 
#                             d_feedforward=d_feedforward, 
#                             input_dim=input_dim)
# 
# output = model(x, adjacency_matrix, graph_metrics)
# print(output)



########################## TestingGraphTransformersAugAttention ########################
import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, ModuleList

class TestingGraphTransformersAugAttention(Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True, dropout=0.1):
        super(TestingGraphTransformersAugAttention, self).__init__()
        self.num_weights = num_weights
        self.use_weights = use_weights
        self.input_linear = Linear(input_dim, d_model)
        self.output_linear = Linear(d_model, input_dim)
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.num_heads = num_heads
        self.d_model = d_model

        # Initialize weight linear layers if using weights
        if self.use_weights:
            self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])
        
        # Attention components (Key-Value-Query matrices)
        self.query_linear = Linear(d_model, d_model)
        self.key_linear = Linear(d_model, d_model)
        self.value_linear = Linear(d_model, d_model)
    
    def forward(self, x, adjacency_matrix, graph_metrics, weights=None):
        # Process adjacency matrix and compute direct, neighborhood, and graph structure scores
        adjacency_matrix = adjacency_matrix.float()
        direct_scores = adjacency_matrix @ x
        neighborhood_similarity = self.compute_neighborhood_similarity(adjacency_matrix, x)
        graph_structure_scores = graph_metrics @ x
        dng_scores = direct_scores + neighborhood_similarity + graph_structure_scores

        # Use weighted linear transformations if weights are provided
        if self.use_weights and weights is not None:
            weighted_x = self.apply_weights(x, weights)
        else:
            weighted_x = self.input_linear(x)

        # Normalize and apply modified attention mechanism
        weighted_x = self.layer_norm(weighted_x)
        x = self.modified_attention(weighted_x, dng_scores)
        x = self.output_linear(x)
        x = self.dropout(x)

        # Combine DNG scores and output for the final result
        final_scores = F.relu(x + dng_scores)
        return final_scores

    def apply_weights(self, x, weights):
        # Apply multiple weight transformations to input
        weighted_x = torch.zeros_like(x)
        for i, weight in enumerate(weights.T):
            weighted_x += self.weight_linears[i](x) * weight.unsqueeze(1)
        return weighted_x

    def modified_attention(self, x, dng_scores):
        # Compute Q, K, V matrices
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Compute scaled attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # Integrate DNG scores into attention mechanism
        attention_scores += dng_scores

        # Normalize scores using softmax and apply to V
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output

    def compute_neighborhood_similarity(self, adjacency_matrix, x):
        # Compute intersection and union for Jaccard similarity
        intersection = adjacency_matrix @ adjacency_matrix
        row_sums = adjacency_matrix.sum(dim=1, keepdim=True)
        col_sums = adjacency_matrix.sum(dim=0, keepdim=True)
        union = row_sums + col_sums - intersection
        similarity = intersection / (union + 1e-6)
        return similarity @ x
    
'''KEY CHANGES:
1.	Weight Application Optimization: Added a separate method apply_weights to modularize and clean up the weight application logic, making it easier to read and maintain.
2.	Attention Mechanism Adjustments:
	•	The modified_attention method is enhanced to properly normalize the attention scores using the scaled dot-product approach.
	•	The integration of dng_scores with attention scores allows the model to factor in graph-specific insights into the attention weights directly.
3.	Compute Neighborhood Similarity: This method is already well-defined, using Jaccard similarity to account for neighborhood overlap, which aligns well with understanding graph structures.
	4.	Modularization: Breaking down the logic into smaller, dedicated methods (apply_weights, modified_attention, compute_neighborhood_similarity) improves the readability and testability of the code.
	5.	Scalability Consideration: For larger graphs, consider using sparse tensors for the adjacency matrix if needed, to handle memory efficiency better.

'''