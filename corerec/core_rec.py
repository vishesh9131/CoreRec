# ###############################################################################################################
#                           --CoreRec: Connecting to the Unseen--
# CoreRec module is designed for graph-based recommendation systems using neural network architectures. It includes:
#     1. GraphTransformer: A neural network model using Transformer architecture for processing graph data.
#     2. GraphDataset: A custom dataset class for handling graph data.
#     3. train_model: A function to train models with options for custom loss functions and training procedures.
#     4. predict: Functions to predict top-k nodes based on model outputs, with optional thresholding.
#     5. draw_graph: A function to visualize graphs with options to highlight top nodes and recommended nodes.
# Note: This module integrates PyTorch for model training and evaluation, and NetworkX for graph manipulation.
# ###############################################################################################################

# Core libraries (previously from common_import)
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import os

# Distributed training
import torch.distributed as dist
from torch.multiprocessing import Process

# Async DDP functions (previously from async_ddp)
from corerec.async_ddp import setup, cleanup

# Models
from corerec.Tmodel import GraphTransformerV2

# Graph convolutions: import from torch_geometric directly if you need them.
# corerec used to ship copies of these layers, which pulled in an undeclared
# torch_geometric dependency and broke `import corerec.core_rec` on a clean install.

# Training and prediction
from corerec.train import train_model
from corerec.predict import predict, explainable_predict
from corerec.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

# Dataset/DataLoader come from torch.utils.data (imported above).

# Optimizers: re-exported from torch.optim.
from torch.optim import (
    ASGD,
    LBFGS,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    NAdam,
    Optimizer,
    RMSprop,
    SparseAdam,
)


# __all__ export list for clean imports
__all__ = [
    # Core libraries
    "np",
    "pd",
    "torch",
    "nx",
    "optim",
    "plt",
    # Data utilities
    "Dataset",
    "DataLoader",
    # Models
    "GraphTransformerV2",
    # Training
    "train_model",
    # Prediction
    "predict",
    "explainable_predict",
    # Metrics
    "jaccard_similarity",
    "adamic_adar_index",
    "aaj_accuracy",
    # Optimizers
    "Adam",
    "NAdam",
    "Adamax",
    "Optimizer",
    "Adadelta",
    "Adagrad",
    "ASGD",
    "LBFGS",
    "RMSprop",
    "SGD",
    "SparseAdam",
    # Distributed training
    "setup",
    "cleanup",
    "dist",
    "Process",
]

# Note: FormatMaster is available separately via:
# from corerec.format_master import ds_format_loader, cr_formatMaster, format_library
