import torch
import torch.nn as nn
import torch.nn.functional as F

class DSSM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dims=[256, 128],
                 dropout=0.5):
        """
        Initialize the DSSM model.

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dims (list): List of hidden layer dimensions.
            dropout (float): Dropout rate.
        """
        super(DSSM, self).__init__()
        
        # Text Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Fully Connected Layers
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.fc = nn.Sequential(*layers)
        
        # Output Embedding Layer
        self.output = nn.Linear(input_dim, hidden_dim)  # Final embedding
        
    def forward(self, text):
        """
        Forward pass of the DSSM model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Semantic embeddings of shape (batch_size, embed_dim).
        """
        # Text Embedding
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        # Fully Connected Layers
        features = self.fc(embedded)  # (batch_size, hidden_dims[-1])
        
        # Output Embedding
        output = self.output(features)  # (batch_size, hidden_dim)
        
        # Normalize embeddings
        output = F.normalize(output, p=2, dim=1)
        
        return output