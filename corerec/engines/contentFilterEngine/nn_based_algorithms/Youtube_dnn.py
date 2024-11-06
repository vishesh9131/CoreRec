import torch
import torch.nn as nn
import torch.nn.functional as F

class YoutubeDNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dims=[256, 128], 
                 dropout=0.5, 
                 num_classes=1):
        """
        Initialize the YouTube Deep Neural Network (YoutubeDNN) model.

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dims (list): List of hidden layer dimensions.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(YoutubeDNN, self).__init__()
        
        # Embedding Layer
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
        
        # Output Layer
        self.output = nn.Linear(input_dim, num_classes)
        
    def forward(self, text):
        """
        Forward pass of the YoutubeDNN model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.mean(dim=1)   # (batch_size, embedding_dim)
        
        # Fully Connected Layers
        features = self.fc(embedded)  # (batch_size, hidden_dims[-1])
        
        # Output Layer
        out = self.output(features)   # (batch_size, num_classes)
        
        return out