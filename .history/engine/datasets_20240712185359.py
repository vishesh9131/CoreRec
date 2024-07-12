"""
datasets.py

This module defines the GraphDataset class, which is a custom dataset class for handling graph data in PyTorch.

Classes:
    GraphDataset: A PyTorch Dataset class for graph data, supporting adjacency and weight matrices.

Usage:
    from engine.datasets import GraphDataset

    # Example usage
    adj_matrix = [[0, 1], [1, 0]]
    weight_matrix = [[0.5, 0.5], [0.5, 0.5]]
    dataset = GraphDataset(adj_matrix, weight_matrix)
    print(len(dataset))  # Output: 2
    print(dataset[0])    # Output: (tensor([0., 1.]), tensor([0.5, 0.5]))
"""

from torch.utils.data import Dataset
import torch

class GraphDataset(Dataset):
    """
    A custom PyTorch Dataset class for handling graph data.

    Attributes:
        adj_matrix (torch.Tensor): The adjacency matrix of the graph.
        weight_matrix (torch.Tensor, optional): The weight matrix of the graph.

    Methods:
        __len__(): Returns the number of nodes in the graph.
        __getitem__(idx): Returns the node features and weights for a given index.
    """
    def __init__(self, adj_matrix, weight_matrix=None):
        """
        Initializes the GraphDataset with adjacency and optional weight matrices.

        Args:
            adj_matrix (list or np.ndarray): The adjacency matrix of the graph.
            weight_matrix (list or np.ndarray, optional): The weight matrix of the graph. Defaults to None.
        """
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        if weight_matrix is not None:
            self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        else:
            self.weight_matrix = None

    def __len__(self):
        """
        Returns the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph.
        """
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        """
        Returns the node features and weights for a given index.

        Args:
            idx (int): The index of the node.

        Returns:
            tuple: A tuple containing the node features and weights.
        """
        node_features = self.adj_matrix[idx]
        if self.weight_matrix is not None:
            weights = self.weight_matrix[idx]
            return node_features, weights
        return node_features, node_features

        """
        Returns the node features and weights for a given index.

        Args:
            idx (int): The index of the node.

        Returns:
            tuple: A tuple containing the node features and weights.
        """