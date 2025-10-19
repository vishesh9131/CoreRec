# dkn.py
import torch
import corerec.torch_nn as nn  # Use custom module instead of torch.nn
import torch.nn.functional as F

def get_divisible_num_heads(embed_dim, max_heads=8):
    """
    Returns the largest number of heads less than or equal to max_heads that divides embed_dim.

    Args:
        embed_dim (int): The embedding dimension.
        max_heads (int): The maximum number of heads to consider.

    Returns:
        int: A suitable number of heads.
    """
    for heads in range(max_heads, 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1  # Fallback to single head if no suitable number found

class DKN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 entity_embedding_dim, 
                 knowledge_graph_size, 
                 text_kernel_sizes=[3, 4, 5], 
                 text_num_filters=100, 
                 dropout=0.5,
                 num_classes=1):
        """
        Initialize the DKN model.

        Args:
            vocab_size (int): Size of the vocabulary for text encoding.
            embedding_dim (int): Dimension of word embeddings.
            entity_embedding_dim (int): Dimension of entity embeddings from the knowledge graph.
            knowledge_graph_size (int): Number of entities in the knowledge graph.
            text_kernel_sizes (list): List of kernel sizes for text CNN.
            text_num_filters (int): Number of filters per kernel size for text CNN.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
        """
        super(DKN, self).__init__()
        
        # Text Embedding Layer
        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # Text CNN Encoder
        self.text_convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=text_num_filters, 
                      kernel_size=k)
            for k in text_kernel_sizes
        ])
        
        # Knowledge Graph Embedding Layer
        self.entity_embedding = nn.Embedding(num_embeddings=knowledge_graph_size, embedding_dim=entity_embedding_dim, padding_idx=0)
        
        # Attention Mechanism
        embed_dim = text_num_filters * len(text_kernel_sizes)  # 100 * 3 = 300
        num_heads = get_divisible_num_heads(embed_dim, max_heads=8)
        print(f"Initializing MultiheadAttention with embed_dim={embed_dim} and num_heads={num_heads}")
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
        
        # Fusion Layer
        self.fusion = nn.Linear(embed_dim + entity_embedding_dim, 256)
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Output Layer
        self.fc_out = nn.Linear(256, num_classes)
        
    def forward(self, text, entities):
        """
        Forward pass of the DKN model.

        Args:
            text (torch.Tensor): Input text tensor of shape (batch_size, seq_length).
            entities (torch.Tensor): Input entity tensor of shape (batch_size, num_entities).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Text Embedding
        embedded_text = self.text_embedding(text)  # (batch_size, seq_length, embedding_dim)
        embedded_text = embedded_text.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # Text CNN Encoder
        text_conv_out = [F.relu(conv(embedded_text)) for conv in self.text_convs]  # List of (batch_size, num_filters, L_out)
        text_pooled = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in text_conv_out]  # List of (batch_size, num_filters)
        text_features = torch.cat(text_pooled, dim=1)  # (batch_size, embed_dim)
        
        # Knowledge Graph Embedding
        entity_embedded = self.entity_embedding(entities)  # (batch_size, num_entities, entity_embedding_dim)
        entity_features = torch.mean(entity_embedded, dim=1)  # (batch_size, entity_embedding_dim)
        
        # Attention Mechanism
        text_features = text_features.unsqueeze(1)  # (batch_size, 1, embed_dim)
        text_features, _ = self.attention(text_features, text_features, text_features)  # (batch_size, 1, embed_dim)
        text_features = text_features.squeeze(1)  # (batch_size, embed_dim)
        
        # Fusion Layer
        fused = torch.cat([text_features, entity_features], dim=1)  # (batch_size, embed_dim + entity_embedding_dim)
        fused = self.fusion(fused)  # (batch_size, 256)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        # Output Layer
        out = self.fc_out(fused)  # (batch_size, num_classes)
        
        return out
    