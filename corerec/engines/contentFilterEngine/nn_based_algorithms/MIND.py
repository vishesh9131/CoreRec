import torch
import torch.nn as nn
import torch.nn.functional as F

class MIND(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 num_interests=4, 
                 interest_dim=64, 
                 dropout=0.5, 
                 num_classes=1):
        """
        Initialize the Multi-Interest Network for Recommendation (MIND).

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            num_interests (int): Number of distinct user interests to capture.
            interest_dim (int): Dimension of each interest representation.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(MIND, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Interest Embedding Layers
        self.interest_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, interest_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_interests)
        ])
        
        # Fusion Layer
        self.fusion = nn.Linear(interest_dim * num_interests, 128)
        
        # Output Layer
        self.fc_out = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        Forward pass of the MIND model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.mean(dim=1)   # (batch_size, embedding_dim)
        
        # Interest Embeddings
        interests = []
        for layer in self.interest_layers:
            interest = layer(embedded)  # (batch_size, interest_dim)
            interests.append(interest)
        interests = torch.cat(interests, dim=1)  # (batch_size, interest_dim * num_interests)
        
        # Fusion Layer
        fused = F.relu(self.fusion(interests))  # (batch_size, 128)
        fused = self.dropout(fused)
        
        # Output Layer
        out = self.fc_out(fused)  # (batch_size, num_classes)
        
        return out