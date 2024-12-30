import torch
from torch import nn
from Tmodel import GraphTransformerV2

class TransformerRecommender:
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Initialize the transformer model
        self.model = GraphTransformerV2(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=num_heads,
            d_feedforward=embedding_dim * 4,
            input_dim=2 * embedding_dim,  # Incorrect input_dim
            dropout=dropout
        )
        
        # User and Item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Initialize adjacency matrix for batch size
        self.batch_adjacency_matrix = None
        self.batch_graph_metrics = None
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters())
    
    def create_batch_graph_structure(self, batch_size):
        """
        Creates batch-specific adjacency matrix and graph metrics
        """
        # Create adjacency matrix for the batch (batch_size x batch_size)
        adj_matrix = torch.zeros((batch_size, batch_size))
        
        # Create basic graph metrics
        graph_metrics = {
            'degree': torch.zeros(batch_size),
            'clustering': torch.zeros(batch_size),
            'centrality': torch.zeros(batch_size)
        }
        
        return adj_matrix, graph_metrics
    
    def update_batch_graph_structure(self, user_ids, item_ids, batch_size):
        """
        Updates the batch-specific adjacency matrix and graph metrics
        """
        # Create new batch-specific adjacency matrix
        adj_matrix = torch.zeros((batch_size, batch_size))
        
        # Create connections between users and items within the batch
        for i in range(batch_size):
            for j in range(batch_size):
                if user_ids[i] == user_ids[j] or item_ids[i] == item_ids[j]:
                    adj_matrix[i, j] = 1.0
        
        # Calculate basic graph metrics for the batch
        graph_metrics = {
            'degree': adj_matrix.sum(dim=1),
            'clustering': torch.zeros(batch_size),  # Simplified clustering coefficient
            'centrality': adj_matrix.sum(dim=0) / batch_size  # Simplified centrality measure
        }
        
        return adj_matrix, graph_metrics
    
    def parameters(self):
        """
        Returns all trainable parameters
        """
        return list(self.model.parameters()) + \
               list(self.user_embeddings.parameters()) + \
               list(self.item_embeddings.parameters())
    def train_step(self, user_ids, item_ids, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = user_ids.size(0)
        
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        
        # Concatenate user and item embeddings
        input_emb = torch.cat([user_emb, item_emb], dim=1)  # Shape: [batch_size, 2*embedding_dim]
        
        # Update batch-specific graph structure
        adj_matrix, graph_metrics = self.update_batch_graph_structure(user_ids, item_ids, batch_size)
        
        # Convert graph_metrics to a tensor
        graph_metrics_tensor = torch.stack([
            graph_metrics['degree'], 
            graph_metrics['clustering'], 
            graph_metrics['centrality']
        ]).T  # Shape: [batch_size, 3]
        
        # Forward pass
        output = self.model(input_emb, adj_matrix, graph_metrics_tensor)
        
        # Calculate prediction (take mean of output)
        pred = output.mean(dim=1)
        
        # Calculate loss
        loss = self.criterion(pred, labels.float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict(self, user_ids, item_ids):
        self.model.eval()
        batch_size = user_ids.size(0)
        
        with torch.no_grad():
            user_emb = self.user_embeddings(user_ids)
            item_emb = self.item_embeddings(item_ids)
            input_emb = torch.cat([user_emb, item_emb], dim=1)
            
            # Create batch-specific graph structure for prediction
            adj_matrix, graph_metrics = self.update_batch_graph_structure(user_ids, item_ids, batch_size)
            
            # Convert graph_metrics to a tensor
            graph_metrics_tensor = torch.stack([
                graph_metrics['degree'], 
                graph_metrics['clustering'], 
                graph_metrics['centrality']
            ]).T  # Shape: [batch_size, 3]
            
            output = self.model(input_emb, adj_matrix, graph_metrics_tensor)
            pred = torch.sigmoid(output.mean(dim=1))
        return pred

# Example usage:
if __name__ == "__main__":
    # Example parameters
    num_users = 1000
    num_items = 500
    batch_size = 64
    
    # Initialize recommender
    recommender = TransformerRecommender(num_users, num_items)
    
    # Example training data
    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Training step
    loss = recommender.train_step(user_ids, item_ids, labels)
    print(f"Training loss: {loss}")
    
    # Make predictions
    test_users = torch.tensor([0, 1, 2])
    test_items = torch.tensor([0, 1, 2])
    predictions = recommender.predict(test_users, test_items)
    print(f"Predictions: {predictions}")
