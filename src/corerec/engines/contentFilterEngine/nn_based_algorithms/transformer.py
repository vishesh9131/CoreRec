# transformer implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.1, num_classes=2):
        """
        Initialize the Transformer model.

        Args:
            input_dim (int): Dimension of the input features.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the feedforward network.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Check if src is 2D and add a sequence dimension if necessary
        if src.dim() == 2:
            src = src.unsqueeze(1)  # (batch_size, seq_length=1, input_dim)
        src = self.embedding(src)  # (batch_size, seq_length, embed_dim)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)  # (seq_length, batch_size, embed_dim)
        memory = memory.mean(dim=0)  # (batch_size, embed_dim)
        memory = self.dropout(memory)
        out = self.fc_out(memory)  # (batch_size, num_classes)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Initialize the positional encoding.

        Args:
            embed_dim (int): Embedding dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Apply positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_length, batch_size, embed_dim).

        Returns:
            torch.Tensor: Positionally encoded tensor.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
