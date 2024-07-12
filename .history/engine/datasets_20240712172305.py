from torch.utils.data import Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self, adj_matrix, weight_matrix=None):
        self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        if weight_matrix is not None:
            self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32)
        else:
            self.weight_matrix = None

    def __len__(self):
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        node_features = self.adj_matrix[idx]
        if self.weight_matrix is not None:
            weights = self.weight_matrix[idx]
            return node_features, weights
        return node_features, node_features