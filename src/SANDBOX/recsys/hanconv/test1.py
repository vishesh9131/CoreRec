import os.path as osp
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB
# from torch_geometric.nn import HANConv
from corerec.cr_pkg.han_conv import HANConv

path = osp.join(osp.dirname(osp.realpath(__file__)), 'src/SANDBOX/dataset/IMDB.csv')
metapaths = [[('movie', 'actor'), ('actor', 'movie')],
             [('movie', 'director'), ('director', 'movie')]]

# Convert sparse matrices to dense before operations
def to_dense(edge_index):
    return edge_index.to_dense() if edge_index.is_sparse else edge_index

# Modify the transform to use dense matrices
class DenseAddMetaPaths(T.BaseTransform):
    def __init__(self, metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True):
        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.drop_unconnected_node_types = drop_unconnected_node_types

    def __call__(self, data):
        # Convert edge indices to dense
        for metapath in self.metapaths:
            for edge_type in metapath:
                if edge_type in data.edge_index_dict:
                    data.edge_index_dict[edge_type] = to_dense(data.edge_index_dict[edge_type])
        return data

transform = DenseAddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)
dataset = IMDB(path, transform=transform)
data = dataset[0]
print(data)

class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out['movie'])
        return out


model = HAN(in_channels=-1, out_channels=3)

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['movie'].train_mask
    loss = F.cross_entropy(out[mask], data['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test() -> List[float]:
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['movie'][split]
        acc = (pred[mask] == data['movie'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


best_val_acc = 0
start_patience = patience = 100
for epoch in range(1, 200):

    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    if best_val_acc <= val_acc:
        patience = start_patience
        best_val_acc = val_acc
    else:
        patience -= 1

    if patience <= 0:
        print('Stopping training as validation accuracy did not improve '
              f'for {start_patience} epochs')
        break

# Function to compute cosine similarity
def cosine_similarity(embeddings, movie_id, top_k=5):
    movie_embedding = embeddings[movie_id].unsqueeze(0)
    similarities = F.cosine_similarity(movie_embedding, embeddings)
    # Get top_k similar movies (excluding the movie itself)
    similar_movies = similarities.argsort(descending=True)[1:top_k+1]
    return similar_movies

# Generate embeddings for all movies
model.eval()
with torch.no_grad():
    # Check if the model output is a dictionary
    out = model(data.x_dict, data.edge_index_dict)
    
    if isinstance(out, dict):
        movie_embeddings = out['movie']
    else:
        # If the output is a tensor, assume it corresponds to movie embeddings
        movie_embeddings = out

# Example: Recommend movies similar to a given movie_id
movie_id = 0  # Replace with the ID of the movie you want recommendations for
similar_movies = cosine_similarity(movie_embeddings, movie_id)

# Print recommended movie IDs
print(f"Movies similar to movie ID {movie_id}: {similar_movies.tolist()}")