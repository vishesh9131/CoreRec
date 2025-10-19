import pytest
import torch
from sage_conv import SAGEConv
from torch_geometric.data import Data

@pytest.fixture
def sample_data():
    # Create a simple graph with 4 nodes and 4 edges
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def test_sageconv_forward(sample_data):
    # Initialize SAGEConv layer
    conv = SAGEConv(in_channels=2, out_channels=2)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions are incorrect"

def test_sageconv_with_different_aggregations(sample_data):
    # Test different aggregation methods
    for aggr in ['mean', 'max', 'sum']:
        conv = SAGEConv(in_channels=2, out_channels=2, aggr=aggr)
        out = conv(sample_data.x, sample_data.edge_index)
        assert out.size() == (4, 2), f"Output dimensions with {aggr} aggregation are incorrect"

def test_sageconv_with_normalization(sample_data):
    # Initialize SAGEConv layer with normalization
    conv = SAGEConv(in_channels=2, out_channels=2, normalize=True)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions with normalization are incorrect"

def test_sageconv_without_root_weight(sample_data):
    # Initialize SAGEConv layer without root weight
    conv = SAGEConv(in_channels=2, out_channels=2, root_weight=False)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions without root weight are incorrect"

def test_sageconv_with_projection(sample_data):
    # Initialize SAGEConv layer with projection
    conv = SAGEConv(in_channels=2, out_channels=2, project=True)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions with projection are incorrect"

def test_sageconv_without_bias(sample_data):
    # Initialize SAGEConv layer without bias
    conv = SAGEConv(in_channels=2, out_channels=2, bias=False)

    # Perform a forward pass
    out = conv(sample_data.x, sample_data.edge_index)

    # Check output dimensions
    assert out.size() == (4, 2), "Output dimensions without bias are incorrect"

if __name__ == "__main__":
    pytest.main([__file__])
