# rnn implementation
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout=0.1, bidirectional=True, num_classes=2):
        """
        Initialize the RNN model.

        Args:
            input_dim (int): Dimension of the input features.
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden state dimension.
            num_layers (int): Number of RNN layers.
            dropout (float): Dropout rate.
            bidirectional (bool): If True, use a bidirectional RNN.
            num_classes (int): Number of output classes.
        """
        super(RNNModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        embed = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        rnn_out, _ = self.rnn(embed)  # (batch_size, seq_length, hidden_dim * num_directions)
        # Use the last hidden state for classification
        if self.rnn.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_hidden = torch.cat(
                (rnn_out[:, -1, :self.rnn.hidden_size], rnn_out[:, 0, self.rnn.hidden_size:]),
                dim=1
            )
        else:
            last_hidden = rnn_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc_out(out)  # (batch_size, num_classes)
        return out
