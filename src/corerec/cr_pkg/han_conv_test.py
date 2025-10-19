import pytest
import torch
from han_conv import HANConv
from torch_geometric.data import HeteroData

@pytest.fixture
def sample_hetero_data():
    # Create a simple heterogeneous graph
    data = HeteroData()

    # Add node types
    data['paper'].x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
    data['author'].x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)

    # Add edge types
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data['author', 'writes', 'paper'].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    return data

def test_hanconv_forward(sample_hetero_data):
    # Define metadata
    metadata = (['paper', 'author'], [('paper', 'cites', 'paper'), ('author', 'writes', 'paper')])

    # Initialize HANConv layer
    conv = HANConv(in_channels={'paper': 2, 'author': 2}, out_channels=4, metadata=metadata)

    # Perform a forward pass
    out_dict = conv(sample_hetero_data.x_dict, sample_hetero_data.edge_index_dict)

    # Check output dimensions
    assert out_dict['paper'].size() == (4, 4), "Output dimensions for 'paper' are incorrect"
    assert out_dict['author'] is None, "Output for 'author' should be None"

def test_hanconv_with_semantic_attention(sample_hetero_data):
    # Define metadata
    metadata = (['paper', 'author'], [('paper', 'cites', 'paper'), ('author', 'writes', 'paper')])

    # Initialize HANConv layer
    conv = HANConv(in_channels={'paper': 2, 'author': 2}, out_channels=4, metadata=metadata)

    # Perform a forward pass with semantic attention weights
    out_dict, semantic_attn_dict = conv(sample_hetero_data.x_dict, sample_hetero_data.edge_index_dict, return_semantic_attention_weights=True)

    # Check output dimensions
    assert out_dict['paper'].size() == (4, 4), "Output dimensions for 'paper' are incorrect"
    assert out_dict['author'] is None, "Output for 'author' should be None"

    # Check semantic attention weights
    assert semantic_attn_dict['paper'] is not None, "Semantic attention weights for 'paper' should not be None"

def test_hanconv_multiple_heads(sample_hetero_data):
    # Define metadata
    metadata = (['paper', 'author'], [('paper', 'cites', 'paper'), ('author', 'writes', 'paper')])

    # Initialize HANConv layer with multiple heads
    conv = HANConv(in_channels={'paper': 2, 'author': 2}, out_channels=4, metadata=metadata, heads=2)

    # Perform a forward pass
    out_dict = conv(sample_hetero_data.x_dict, sample_hetero_data.edge_index_dict)

    # Check output dimensions
    assert out_dict['paper'].size() == (4, 4), "Output dimensions for 'paper' with multiple heads are incorrect"

def test_hanconv_dropout(sample_hetero_data):
    # Define metadata
    metadata = (['paper', 'author'], [('paper', 'cites', 'paper'), ('author', 'writes', 'paper')])

    # Initialize HANConv layer with dropout
    conv = HANConv(in_channels={'paper': 2, 'author': 2}, out_channels=4, metadata=metadata, dropout=0.5)

    # Set the layer to training mode to apply dropout
    conv.train()

    # Perform a forward pass
    out_dict = conv(sample_hetero_data.x_dict, sample_hetero_data.edge_index_dict)

    # Check output dimensions
    assert out_dict['paper'].size() == (4, 4), "Output dimensions for 'paper' with dropout are incorrect"

if __name__ == "__main__":
    pytest.main([__file__])
