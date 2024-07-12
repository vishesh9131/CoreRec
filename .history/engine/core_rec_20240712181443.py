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
from common_import import *
from async_ddp import *

from engine.models import GraphTransformer
# from engine.Tmodel import GraphTransformer

from engine.datasets import GraphDataset
from engine.train import train_model
from engine.predict import predict, explainable_predict
from engine.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

# EXTRAS : visualization functions 
# from engine.vish_graphs import draw_graph, draw_graph_3d

