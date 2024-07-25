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

# This is the Core of your model
from engine.models import *
from engine.Tmodel import GraphTransformerV2
from engine.cr_pkg.gat_conv import *
from engine.cr_pkg.gcn_conv import *
from engine.cr_pkg.han_conv import *
from engine.cr_pkg.sage_conv import *


# In Emergence this will act as Organs to your model
from engine.train import train_model
from engine.predict import predict, explainable_predict
from engine.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

# Importing dataloaders,dataset
from engine.cr_utility.dataloader import *
from engine.cr_utility.dataset import *    #dont call this its not working rn use GraphDataset ookk

# Importing Boosters AKA Optimizers (Note: _funtional and _init_ is commented)
from engine.cr_boosters.adam import *
from engine.cr_boosters.nadam import *
from engine.cr_boosters.adamax import *
from engine.cr_boosters.optimizer import *
from engine.cr_boosters.adadelta import *
from engine.cr_boosters.adagrad import *
from engine.cr_boosters.asgd import *
from engine.cr_boosters.lbfgs import *
from engine.cr_boosters.rmsprop import *
from engine.cr_boosters.sgd import *
from engine.cr_boosters.sparse_adam import *


#Promoted this script to engine.cr_utility.dataset
from engine.datasets import GraphDataset