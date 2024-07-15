import sys
import os
import unittest

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now you can import the modules from the parent directory
import core_rec as cs
import vish_graphs as vg
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# from USECASES.custommodel import SimpleNN

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
        self.dataset = cs.GraphDataset(self.adj_matrix)
        self.data_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.num_epochs = 1

    def test_forward(self):
        x = torch.rand(1, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (1, self.output_dim))

    def test_training(self):
        cs.train_model(self.model, self.data_loader, self.criterion, self.optimizer, self.num_epochs)
        self.assertTrue(True)  # If training completes without error, the test passes

if __name__ == '__main__':
    unittest.main()