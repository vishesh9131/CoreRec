# cnn implementation
import torch
import corerec.torch_nn as nn 

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes, emb_dim=128, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        """
        Initialize the CNN model for classification.

        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
            emb_dim (int): Embedding dimension.
            kernel_sizes (list): List of kernel sizes for convolution.
            num_filters (int): Number of filters per kernel size.
            dropout (float): Dropout rate.
        """
        super(CNN, self).__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.embedding(x)  # (batch_size, emb_dim)
        x = x.unsqueeze(2)  # (batch_size, emb_dim, 1)
        
        # Determine the required padding based on the largest kernel size
        max_kernel_size = max([conv.kernel_size[0] for conv in self.convs])
        pad_size = (max_kernel_size // 2, max_kernel_size // 2)
        x = torch.nn.functional.pad(x, pad_size)  # (batch_size, emb_dim, padded_length)
        
        # Apply convolution and activation
        conv_out = [torch.relu(conv(x)) for conv in self.convs]  # List of (batch_size, num_filters, L)
        
        # Apply max pooling over the time dimension
        pooled = [torch.max(feature_map, dim=2)[0] for feature_map in conv_out]  # List of (batch_size, num_filters)
        
        # Concatenate pooled features
        concat = torch.cat(pooled, dim=1)  # (batch_size, num_filters * len(kernel_sizes))
        
        # Apply dropout
        drop = self.dropout(concat)
        
        # Final fully connected layer
        out = self.fc(drop)  # (batch_size, num_classes)
        
        return out