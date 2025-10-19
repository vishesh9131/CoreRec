import pytest
import torch
from gat_conv import GATConv
from torch_geometric.data import Data

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit.frontend")

@pytest.fixture
def sample_data():
    # Create a simple graph with 4 nodes and 4 edges
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
    edge_attr = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def test_gatconv_forward(sample_data):
    # Initialize GATConv layer
    conv = GATConv(in_channels=2, out_channels=2, heads=2, concat=True)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 4), "Output dimensions are incorrect"

def test_gatconv_with_edge_features(sample_data):
    # Initialize GATConv layer with edge features
    conv = GATConv(in_channels=2, out_channels=2, heads=2, concat=True, edge_dim=1)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index, sample_data.edge_attr)

    # Check output dimensions
    assert out.size() == (4, 4), "Output dimensions with edge features are incorrect"

def test_gatconv_no_self_loops(sample_data):
    # Initialize GATConv layer without self-loops
    conv = GATConv(in_channels=2, out_channels=2, heads=2, concat=True, add_self_loops=False)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 4), "Output dimensions without self-loops are incorrect"

def test_gatconv_dropout(sample_data):
    # Initialize GATConv layer with dropout
    conv = GATConv(in_channels=2, out_channels=2, heads=2, concat=True, dropout=0.5)

    # Set the layer to training mode to apply dropout
    conv.train()

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 4), "Output dimensions with dropout are incorrect"

if __name__ == "__main__":
    pytest.main([__file__])
