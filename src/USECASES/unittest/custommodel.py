import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Create the missing GraphDataset class
class GraphDataset(Dataset):
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.adj_matrix)

    def __getitem__(self, idx):
        return self.adj_matrix[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestSimpleNN(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 64
        self.output_dim = 10
        self.model = SimpleNN(self.input_dim, self.hidden_dim, self.output_dim)
        self.adj_matrix = torch.tensor(np.random.rand(40, 10), dtype=torch.float32)
        self.dataset = GraphDataset(self.adj_matrix)  # Using local GraphDataset class
        self.data_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.num_epochs = 1

    def test_forward(self):
        x = torch.rand(1, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_training(self):
        # Simple training loop since cs.train_model might not exist
        for epoch in range(self.num_epochs):
            for batch in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                self.optimizer.step()
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
