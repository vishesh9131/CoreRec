import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Scoreformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_feedforward, input_dim, num_weights=10, use_weights=True, dropout=0.1):
        super(Scoreformer, self).__init__()
        self.num_weights = num_weights
        self.use_weights = use_weights
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_linear = Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_linear.weight, gain=1.0)
        
        self.dng_projection = Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.dng_projection.weight, gain=1.0)
        
        self.encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.pre_output = Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.pre_output.weight, gain=1.0)
        
        self.output_linear = Linear(d_model, 1)
        nn.init.xavier_uniform_(self.output_linear.weight, gain=1.0)
        
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        
        if self.use_weights:
            self.weight_linears = ModuleList([Linear(input_dim, d_model) for _ in range(num_weights)])
            for layer in self.weight_linears:
                nn.init.xavier_uniform_(layer.weight, gain=1.0)

    def compute_neighborhood_similarity(self, adjacency_matrix, x):
        binary_adj = (adjacency_matrix > 0).float()
        intersection = binary_adj @ binary_adj.T
        row_sums = binary_adj.sum(dim=1, keepdim=True)
        col_sums = binary_adj.sum(dim=0, keepdim=True)
        union = row_sums + col_sums.T - intersection
        similarity = intersection / (union + 1e-8)
        return similarity @ x

    def project_graph_metrics(self, graph_metrics, target_dim):
        if graph_metrics.size(1) < target_dim:
            repeats = (target_dim + graph_metrics.size(1) - 1) // graph_metrics.size(1)
            graph_metrics = graph_metrics.repeat(1, repeats)[:, :target_dim]
        elif graph_metrics.size(1) > target_dim:
            graph_metrics = graph_metrics[:, :target_dim]
        return graph_metrics

    def forward(self, x, adjacency_matrix, graph_metrics, weights=None):
        adjacency_matrix = adjacency_matrix.float()
        graph_metrics = graph_metrics.float()
        batch_size, input_dim = x.shape
        
        # Direct connections
        direct_scores = adjacency_matrix @ x
        
        # Neighborhood similarity
        neighborhood_similarity = self.compute_neighborhood_similarity(adjacency_matrix, x)
        
        # Graph structure scores
        graph_metrics_projected = self.project_graph_metrics(graph_metrics, input_dim)
        graph_structure_scores = graph_metrics_projected * x

        # Combine DNG scores and project
        dng_scores = direct_scores + neighborhood_similarity + graph_structure_scores
        dng_scores = self.dng_projection(dng_scores)
        
        # Process input through transformer
        if self.use_weights and weights is not None:
            weighted_x = torch.zeros(x.size(0), self.d_model, device=x.device)
            for i, weight in enumerate(weights.T):
                projected_x = self.weight_linears[i](x)
                weighted_x += projected_x * weight.unsqueeze(1)
            transformer_input = weighted_x
        else:
            transformer_input = self.input_linear(x)

        transformer_input = self.layer_norm(transformer_input)
        transformer_output = self.transformer_encoder(transformer_input.unsqueeze(1)).squeeze(1)
        
        # Combine and process final output
        combined = transformer_output + dng_scores
        combined = self.dropout(combined)
        output = self.pre_output(combined)
        output = F.relu(output)
        output = self.output_linear(output)
        output = torch.sigmoid(output)
        
        return output.squeeze(-1)

def print_tensor_info(name, tensor):
    print(f"\n{'-'*50}")
    print(f"{name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Values:\n{tensor.detach().numpy().round(3)}")
    print(f"{'-'*50}")

def test_scoreformer_flow():
    # 1. Initialize Model
    print("\n🔧 Initializing Model...")
    model = Scoreformer(
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_feedforward=128,
        input_dim=3,
        num_weights=3,
        use_weights=True,
        dropout=0.1
    )
    
    # 2. Prepare Input Data
    print("\n📊 Preparing Input Data...")
    
    # User features [sports, art, music]
    x = torch.tensor([
        [1.0, 0.0, 1.0],  # Alice: likes sports & music
        [1.0, 1.0, 0.0],  # Bob: likes sports & art
        [0.0, 1.0, 1.0],  # Charlie: likes art & music
        [1.0, 0.0, 0.0],  # Diana: likes sports only
    ])
    print_tensor_info("User Features (x)", x)

    # Connection matrix
    adjacency_matrix = torch.tensor([
        [0.0, 1.0, 0.0, 1.0],  # Alice follows Bob and Diana
        [1.0, 0.0, 1.0, 0.0],  # Bob follows Alice and Charlie
        [0.0, 1.0, 0.0, 1.0],  # Charlie follows Bob and Diana
        [1.0, 0.0, 1.0, 0.0],  # Diana follows Alice and Charlie
    ])
    print_tensor_info("Adjacency Matrix", adjacency_matrix)

    # User metrics [follower_ratio, post_frequency, engagement_rate]
    graph_metrics = torch.tensor([
        [0.8, 0.6, 0.7],  # Alice's metrics
        [0.9, 0.5, 0.6],  # Bob's metrics
        [0.7, 0.8, 0.5],  # Charlie's metrics
        [0.6, 0.7, 0.8],  # Diana's metrics
    ])
    print_tensor_info("Graph Metrics", graph_metrics)

    # Feature weights
    weights = torch.tensor([
        [0.7, 0.2, 0.1],  # Weights for each feature
        [0.7, 0.2, 0.1],
        [0.7, 0.2, 0.1],
        [0.7, 0.2, 0.1],
    ])
    print_tensor_info("Feature Weights", weights)

    # 3. Process through model components
    print("\n🔄 Processing through model...")
    
    with torch.no_grad():
        # Direct connections
        direct_scores = adjacency_matrix @ x
        print_tensor_info("Direct Connection Scores", direct_scores)
        
        # Neighborhood similarity
        binary_adj = (adjacency_matrix > 0).float()
        intersection = binary_adj @ binary_adj.T
        row_sums = binary_adj.sum(dim=1, keepdim=True)
        col_sums = binary_adj.sum(dim=0, keepdim=True)
        union = row_sums + col_sums.T - intersection
        similarity = intersection / (union + 1e-8)
        neighborhood_scores = similarity @ x
        print_tensor_info("Neighborhood Similarity Scores", neighborhood_scores)
        
        # Graph structure scores
        graph_structure_scores = graph_metrics * x
        print_tensor_info("Graph Structure Scores", graph_structure_scores)
        
        # Final model output
        final_scores = model(x, adjacency_matrix, graph_metrics, weights)
        print_tensor_info("Final Recommendation Scores", final_scores)

    # 4. Analyze Results
    print("\n📈 Final Analysis:")
    users = ['Alice', 'Bob', 'Charlie', 'Diana']
    
    for i, user in enumerate(users):
        print(f"\n{user}'s Network Analysis:")
        print("Current connections:", end=" ")
        connections = [users[j] for j in range(len(users)) if adjacency_matrix[i][j] > 0]
        print(", ".join(connections))
        
        print("Potential recommendations:")
        potential_recs = [(users[j], final_scores[j].item()) 
                         for j in range(len(users)) 
                         if j != i and adjacency_matrix[i][j] == 0]
        potential_recs.sort(key=lambda x: x[1], reverse=True)
        
        for rec, score in potential_recs:
            print(f"- {rec} (score: {score:.3f})")
            # Add explanation based on features
            explanations = []
            if torch.dot(x[i], x[users.index(rec)]) > 0:
                common_interests = []
                if x[i][0] == 1 and x[users.index(rec)][0] == 1:
                    common_interests.append("sports")
                if x[i][1] == 1 and x[users.index(rec)][1] == 1:
                    common_interests.append("art")
                if x[i][2] == 1 and x[users.index(rec)][2] == 1:
                    common_interests.append("music")
                if common_interests:
                    explanations.append(f"Common interests: {', '.join(common_interests)}")
            
            mutual_friends = len(set(connections) & 
                               set([users[j] for j in range(len(users)) 
                                   if adjacency_matrix[users.index(rec)][j] > 0]))
            if mutual_friends > 0:
                explanations.append(f"{mutual_friends} mutual connection(s)")
            
            if explanations:
                print(f"  Why? {'; '.join(explanations)}")

if __name__ == "__main__":
    test_scoreformer_flow()