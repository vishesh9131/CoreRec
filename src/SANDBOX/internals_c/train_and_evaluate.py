import numpy as np
import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from interface import gcn_layer

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Convert data to numpy arrays
x = data.x.numpy()
adj = np.eye(x.shape[0]) + data.edge_index.numpy()  # Add self-loops
w = np.random.randn(x.shape[1], 16).astype(np.float32)  # Random weights

# Define optimizer and training loop
optimizer = optim.Adam([torch.tensor(w, requires_grad=True)], lr=0.01)

def train(epochs=200):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = gcn_layer(x, adj, w)
        loss = torch.nn.functional.nll_loss(torch.tensor(out[data.train_mask]), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate model
def evaluate():
    out = gcn_layer(x, adj, w)
    _, pred = torch.tensor(out).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc

# Train and evaluate
train()
accuracy = evaluate()
print(f'Accuracy: {accuracy:.4f}')