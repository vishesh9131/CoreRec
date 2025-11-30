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

# Graph convolutions
from corerec.cr_pkg.gat_conv import GATConv
from corerec.cr_pkg.gcn_conv import GCNConv
from corerec.cr_pkg.han_conv import HANConv
from corerec.cr_pkg.sage_conv import SAGEConv

# Training and prediction
from corerec.train import train_model
from corerec.predict import predict, explainable_predict
from corerec.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

# Data utilities
from corerec.cr_utility.dataloader import DataLoader

# Note: GraphDataset may need to be imported differently depending on
# implementation
try:
    from corerec.cr_utility.dataset import GraphDataset
except ImportError:
    # GraphDataset might be defined elsewhere or needs custom implementation
    GraphDataset = None

# Optimizers (Boosters)
from corerec.cr_boosters.adam import Adam
from corerec.cr_boosters.nadam import NAdam
from corerec.cr_boosters.adamax import Adamax
from corerec.cr_boosters.optimizer import Optimizer
from corerec.cr_boosters.adadelta import Adadelta
from corerec.cr_boosters.adagrad import Adagrad
from corerec.cr_boosters.asgd import ASGD
from corerec.cr_boosters.lbfgs import LBFGS
from corerec.cr_boosters.rmsprop import RMSprop
from corerec.cr_boosters.sgd import SGD
from corerec.cr_boosters.sparse_adam import SparseAdam


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
    "GraphDataset",
    # Models
    "GraphTransformerV2",
    # Graph convolutions
    "GATConv",
    "GCNConv",
    "HANConv",
    "SAGEConv",
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
