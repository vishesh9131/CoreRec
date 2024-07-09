import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch_geometric.lo

import sys
sys.path.append('/Users/visheshyadav/Documents/GitHub/CoreRec/engine')
from core_rec import GraphTransformer, train_model, predict

import sys
sys.path.append('/Users/visheshyadav/Documents/GitHub/CoreRec/engine/torch_nn')
from torch_nn import *

# Load your data
labels_df = pd.read_csv('SANDBOX/Analysis/data_mother/500label.csv')
labels = labels_df['Names'].tolist()
label_to_index = {label: idx for idx, label in enumerate(labels)}

# Load adjacency matrix
# adj_matrix = pd.read_csv('label.csv', header=None).values
adj_matrix=np.loadtxt('SANDBOX/Analysis/data_mother/wgtlabel.csv', delimiter=',')

# Create edge index for PyTorch Geometric
edge_index = torch_geometric.utils.dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))[0]

# Create node features (for simplicity, using identity matrix)
num_nodes = adj_matrix.shape[0]
x = torch.eye(num_nodes, dtype=torch.float)

# Create labels (for simplicity, using node indices as labels)
y = torch.tensor(range(num_nodes), dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_nodes, 16)
        self.conv2 = GCNConv(16, num_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define GAT model
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_nodes, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16 * 8, num_nodes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_nodes, 16)
        self.conv2 = SAGEConv(16, num_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Placeholder for other models
class TransE(torch.nn.Module):
    def __init__(self):
        super(TransE, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class TransR(torch.nn.Module):
    def __init__(self):
        super(TransR, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class DistMult(torch.nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class ComplEx(torch.nn.Module):
    def __init__(self):
        super(ComplEx, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class HAN(torch.nn.Module):
    def __init__(self):
        super(HAN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class MetaPath2Vec(torch.nn.Module):
    def __init__(self):
        super(MetaPath2Vec, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class GCF(torch.nn.Module):
    def __init__(self):
        super(GCF, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class GRMF(torch.nn.Module):
    def __init__(self):
        super(GRMF, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class STAGE(torch.nn.Module):
    def __init__(self):
        super(STAGE, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class SRGNN(torch.nn.Module):
    def __init__(self):
        super(SRGNN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class DeepWalk(torch.nn.Module):
    def __init__(self):
        super(DeepWalk, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

class Node2Vec(torch.nn.Module):
    def __init__(self):
        super(Node2Vec, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, 16)
        self.linear = torch.nn.Linear(16, num_nodes)  # Add a linear layer

    def forward(self, data):
        x = self.embedding(data.x.argmax(dim=1))
        return self.linear(x)

# Define MetaExploitModel
class MetaExploitModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(MetaExploitModel, self).__init__()
        num_layers = 1
        d_model = 128
        num_heads = 2
        d_feedforward = 512
        self.model = GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim, use_weights=True)

    def forward(self, data):
        adj_matrix = data.x.numpy()  # Assuming data.x is the adjacency matrix
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        print(f"adj_matrix shape: {adj_matrix.shape}")
        output = self.model(adj_matrix)
        return output


# Define batch size
batch_size = 32

# Create DataLoader for mini-batch training
data_list = [data]  # Assuming 'data' is a single Data object, wrap it in a list
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)


# Placeholder for model evaluation
def evaluate_model(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    y_true = data.y.numpy()
    y_pred = pred.numpy()
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    y_true_binarized = label_binarize(y_true, classes=np.arange(num_nodes))
    y_pred_binarized = label_binarize(y_pred, classes=np.arange(num_nodes))
    
    roc_auc = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn = np.diag(cm)
    fp = cm.sum(axis=0) - tn
    fn = cm.sum(axis=1) - tn
    tp = cm.sum() - (fp + fn + tn)
    
    # Debugging print statements
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = np.mean(np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0))
        sensitivity = np.mean(np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "roc_auc": roc_auc,
        "mcc": mcc
    }

# List of models to benchmark
models_to_benchmark = {
    "CoreRec": MetaExploitModel(input_dim=adj_matrix.shape[1]) ,
    "GCN": GCN(),
    "GraphSAGE": GraphSAGE(),
    "TransE": TransE(),
    "TransR": TransR(),
    "DistMult": DistMult(),
    "ComplEx": ComplEx(),
    "HAN": HAN(),
    "MetaPath2Vec": MetaPath2Vec(),
    "GCF": GCF(),
    "GRMF": GRMF(),
    "GAT": GAT(),
    "STAGE": STAGE(),
    "SR-GNN": SRGNN(),
    "DeepWalk": DeepWalk(),
    "Node2Vec": Node2Vec()
}

# Dictionary to store benchmark results
benchmark_results = {}

# Train and evaluate each model
for model_name, model in models_to_benchmark.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    print(f"Training {model_name}...")
    accumulation_steps = 4  # Number of steps to accumulate gradients
    for epoch in range(50):  # Reduce the number of epochs
        optimizer.zero_grad()
        for batch in loader:
            out = model(batch)
            if out is None:
                print(f"Model {model_name} returned None output")
                continue
            if out.shape != (num_nodes, num_nodes):
                print(f"Unexpected output shape for {model_name}: {out.shape}")
                continue
            loss = F.nll_loss(out, batch.y)  # Removed train_mask for simplicity
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            loss.backward()
            
            if (epoch + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = F.nll_loss(out, data.y)  # Removed val_mask for simplicity
        print(f"Validation Loss for {model_name}: {val_loss.item()}")
    
    metrics = evaluate_model(model, data)
    print(f"Metrics for {model_name}: {metrics}")
    benchmark_results[model_name] = metrics

# Convert results to DataFrame for easier plotting
df = pd.DataFrame(benchmark_results).T

# Plot the results with padding between models
sns.set(style="whitegrid")
palette = sns.color_palette("Set2")

fig, ax = plt.subplots(figsize=(14, 10))
x = np.arange(len(models_to_benchmark))
width = 0.08  # Reduce the width of the bars to add padding

metrics = df.columns
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df[metric], width, label=metric, capsize=5, color=palette[i % len(palette)])

ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Benchmark Results Comparison', fontsize=16, weight='bold')
ax.set_xticks(x + width * (len(metrics) - 1) / 2)
ax.set_xticklabels(models_to_benchmark.keys(), rotation=90, fontsize=12)
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()