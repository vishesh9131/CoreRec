import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from corerec.async_ddp import worker, ParameterServer
from multiprocessing import Process, freeze_support
import numpy as np
import vish_graphs as vg
import core_rec as cs
from common_import import *
from corerec.async_ddp import find_free_port  #dpp= distributed data processing
import os

num_people = 10
file_path = vg.generate_random_graph(num_people, file_path="blobbag.csv", seed=56)
adj_matrix = np.loadtxt(file_path, delimiter=",")

graph_dataset = cs.GraphDataset(adj_matrix)
data_loader = DataLoader(graph_dataset, batch_size=5, shuffle=True)

num_layers = 2
d_model = 128
num_heads = 8
d_feedforward = 512
input_dim = len(adj_matrix[0])
model = cs.GraphTransformer(num_layers, d_model, num_heads, d_feedforward, input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
world_size = 4  # Total processes: 1 parameter server + 3 workers

    # Start the parameter server
port = find_free_port()  # Find a free port
os.environ['MASTER_PORT'] = str(port)  # Set the port in the environment
ps = Process(target=ParameterServer(model, world_size).run)
ps.start()

processes = []
for rank in range(1, world_size):  # Worker ranks from 1 to world_size-1
    p = Process(target=worker, args=(rank, model, data_loader, optimizer, num_epochs, world_size, port))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
    ps.join()

num_workers = 4
# Create a ParameterServer instance
parameter_server = ParameterServer(model, num_workers)

# Start the parameter server
parameter_server.run()

# Start the workers
for rank in range(1, num_workers + 1):
    port = 12345  # Use a fixed port for simplicity
    worker(rank, model, data_loader, optimizer, num_epochs, num_workers, port)