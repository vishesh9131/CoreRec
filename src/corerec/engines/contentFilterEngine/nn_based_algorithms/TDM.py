import torch
import torch.nn as nn
import torch.nn.functional as F

class TDM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim=128, 
                 dropout=0.5, 
                 num_classes=1):
        """
        Initialize the Text Domain Model (TDM).

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(TDM, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Convolutional Layer for Domain Features
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        """
        Forward pass of the TDM model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Embedding
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # Convolution and Pooling
        conv_out = F.relu(self.conv(embedded))  # (batch_size, hidden_dim, seq_length)
        pooled = self.pool(conv_out).squeeze(2)  # (batch_size, hidden_dim)
        
        # Fully Connected Layer
        out = self.fc(pooled)  # (batch_size, num_classes)
        out = self.dropout(out)
        
        return out