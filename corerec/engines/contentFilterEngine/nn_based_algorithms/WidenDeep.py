import torch
import torch.nn as nn
import torch.nn.functional as F

class WidenDeep(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dims=[128, 64], 
                 dropout=0.5, 
                 num_classes=1):
        """
        Initialize the Wide & Deep model.

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dims (list): List of hidden layer dimensions for the deep part.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(WidenDeep, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Wide (Linear) Component
        self.wide = nn.Linear(embedding_dim, num_classes)
        
        # Deep Component
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(input_dim, num_classes)
        
    def forward(self, text):
        """
        Forward pass of the Wide & Deep model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        # Wide Component
        wide_out = self.wide(embedded)  # (batch_size, num_classes)
        
        # Deep Component
        deep_features = self.deep(embedded)  # (batch_size, hidden_dims[-1])
        deep_out = self.deep_output(deep_features)  # (batch_size, num_classes)
        
        # Combine Wide and Deep
        out = wide_out + deep_out  # (batch_size, num_classes)
        
        return out