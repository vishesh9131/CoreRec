import pytest
import torch
from gcn_conv import GCNConv
from torch_geometric.data import Data

@pytest.fixture
def sample_data():
    # Create a simple graph with 4 nodes and 4 edges
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
    edge_weight = torch.tensor([1, 1, 1, 1], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

def test_gcnconv_forward(sample_data):
    # Initialize GCNConv layer
    conv = GCNConv(in_channels=2, out_channels=2)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions are incorrect"

def test_gcnconv_with_edge_weights(sample_data):
    # Initialize GCNConv layer
    conv = GCNConv(in_channels=2, out_channels=2)

    # Perform a forward pass with edge weights
    out = conv(sample_data.x, sample_data.edge_index, sample_data.edge_weight)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions with edge weights are incorrect"

def test_gcnconv_no_self_loops(sample_data):
    # Initialize GCNConv layer without self-loops
    conv = GCNConv(in_channels=2, out_channels=2, add_self_loops=False)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions without self-loops are incorrect"

def test_gcnconv_cached(sample_data):
    # Initialize GCNConv layer with caching
    conv = GCNConv(in_channels=2, out_channels=2, cached=True)

    # Perform a forward pass
    out1 = conv(sample_data.x, sample_data.edge_index)
    out2 = conv(sample_data.x, sample_data.edge_index)

    # Check that the outputs are the same, indicating caching is working
    assert torch.allclose(out1, out2), "Caching is not working correctly"

def test_gcnconv_improved(sample_data):
    # Initialize GCNConv layer with improved normalization
    conv = GCNConv(in_channels=2, out_channels=2, improved=True)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions with improved normalization are incorrect"

def test_gcnconv_bias(sample_data):
    # Initialize GCNConv layer without bias
    conv = GCNConv(in_channels=2, out_channels=2, bias=False)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions without bias are incorrect"

if __name__ == "__main__":
    pytest.main([__file__])
