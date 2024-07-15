import numpy as np
import ctypes
from scipy.sparse import coo_matrix

# Load the shared library
gat = ctypes.CDLL('./gat.so')
graph_transformer = ctypes.CDLL('./graph_transformer.so')

# Define the argument and return types for GAT
gat.gat_layer.argtypes = [ctypes.POINTER(ctypes.c_float), 
                          ctypes.POINTER(ctypes.c_float), 
                          ctypes.POINTER(ctypes.c_float), 
                          ctypes.POINTER(ctypes.c_float), 
                          ctypes.POINTER(ctypes.c_float), 
                          ctypes.POINTER(ctypes.c_float), 
                          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]

# Define the argument and return types for Graph Transformer
graph_transformer.graph_transformer_layer.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.POINTER(ctypes.c_float), 
                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Function to call the C GAT layer
def gat_layer(x, adj, a_src, a_dst, w, alpha=0.2):
    num_nodes, input_dim = x.shape
    output_dim = w.shape[1]
    
    x = x.astype(np.float32)
    adj = adj.astype(np.float32)
    a_src = a_src.astype(np.float32)
    a_dst = a_dst.astype(np.float32)
    w = w.astype(np.float32)
    
    out = np.zeros((num_nodes, output_dim), dtype=np.float32)
    
    gat.gat_layer(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  adj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  a_src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  a_dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                  num_nodes, input_dim, output_dim, alpha)
    return out

def graph_transformer_layer(x, adj, w_q, w_k, w_v, w_out):
    num_nodes, input_dim = x.shape
    output_dim = w_q.shape[1]   
    x = x.astype(np.float32)
    adj = adj.astype(np.float32)
    w_q = w_q.astype(np.float32)
    w_k = w_k.astype(np.float32)
    w_v = w_v.astype(np.float32)
    w_out = w_out.astype(np.float32)

    out = np.zeros((num_nodes, output_dim), dtype=np.float32)

    graph_transformer.graph_transformer_layer(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          adj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          w_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          w_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          w_v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          w_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          num_nodes, input_dim, output_dim)

    return out

### Training and Evaluation for GAT and Graph Transformer

import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Convert data to numpy arrays
x = data.x.numpy()
edge_index = data.edge_index.numpy()
num_nodes = x.shape[0]

# Create adjacency matrix
adj = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)).toarray()
adj += np.eye(num_nodes)  # Add self-loops

w = np.random.randn(x.shape[1], 16).astype(np.float32)  # Random weights
a_src = np.random.randn(x.shape[0], 1).astype(np.float32)  # Random attention weights
a_dst = np.random.randn(x.shape[0], 1).astype(np.float32)

# Define optimizer and training loop for GAT
w_tensor = torch.tensor(w, requires_grad=True)
optimizer_gat = optim.Adam([w_tensor], lr=0.01)

def train_gat(epochs=200):
    for epoch in range(epochs):
        optimizer_gat.zero_grad()
        out = gat_layer(x, adj, a_src, a_dst, w_tensor.detach().numpy())
        out_tensor = torch.tensor(out, requires_grad=True)
        loss = torch.nn.functional.nll_loss(out_tensor[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_tensor], max_norm=1.0)  # Gradient clipping
        optimizer_gat.step()
        print(f'GAT Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate GAT model
def evaluate_gat():
    out = gat_layer(x, adj, a_src, a_dst, w_tensor.detach().numpy())
    _, pred = torch.tensor(out).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc

# Training and evaluation for Graph Transformer
w_q = np.random.randn(x.shape[1], 16).astype(np.float32)
w_k = np.random.randn(x.shape[1], 16).astype(np.float32)
w_v = np.random.randn(x.shape[1], 16).astype(np.float32)
w_out = np.random.randn(16, 16).astype(np.float32)
w_q_tensor = torch.tensor(w_q, requires_grad=True)
w_k_tensor = torch.tensor(w_k, requires_grad=True)
w_v_tensor = torch.tensor(w_v, requires_grad=True)
w_out_tensor = torch.tensor(w_out, requires_grad=True)
optimizer_gt = optim.Adam([w_q_tensor, w_k_tensor, w_v_tensor, w_out_tensor], lr=0.01)

def train_graph_transformer(epochs=200):
    for epoch in range(epochs):
        optimizer_gt.zero_grad()
        out = graph_transformer_layer(x, adj, w_q_tensor.detach().numpy(), w_k_tensor.detach().numpy(), w_v_tensor.detach().numpy(), w_out_tensor.detach().numpy())
        out_tensor = torch.tensor(out, requires_grad=True)
        loss = torch.nn.functional.nll_loss(out_tensor[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w_q_tensor, w_k_tensor, w_v_tensor, w_out_tensor], max_norm=1.0)  # Gradient clipping
        optimizer_gt.step()
        print(f'Graph Transformer Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate Graph Transformer model
def evaluate_graph_transformer():
    out = graph_transformer_layer(x, adj, w_q_tensor.detach().numpy(), w_k_tensor.detach().numpy(), w_v_tensor.detach().numpy(), w_out_tensor.detach().numpy())
    _, pred = torch.tensor(out).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    return acc

# Train and evaluate models
train_gat()
gat_accuracy = evaluate_gat()
print(f'GAT Accuracy: {gat_accuracy:.4f}')

train_graph_transformer()
gt_accuracy = evaluate_graph_transformer()
print(f'Graph Transformer Accuracy: {gt_accuracy:.4f}')