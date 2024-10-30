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
from corerec.common_import import *
from corerec.async_ddp import *

# This is the Core of your model
from corerec.models import *
from corerec.Tmodel import GraphTransformerV2
from corerec.cr_pkg.gat_conv import *
from corerec.cr_pkg.gcn_conv import *
from corerec.cr_pkg.han_conv import *
from corerec.cr_pkg.sage_conv import *


# In Emergence this will act as Organs to your model
from corerec.train import train_model
from corerec.predict import predict, explainable_predict
from corerec.metrics import jaccard_similarity, adamic_adar_index, aaj_accuracy

# Importing dataloaders,dataset
from corerec.cr_utility.dataloader import *
from corerec.cr_utility.dataset import *    #dont call this its not working rn use GraphDataset ookk

# Importing Boosters AKA Optimizers (Note: _funtional and _init_ is commented)
from corerec.cr_boosters.adam import *
from corerec.cr_boosters.nadam import *
from corerec.cr_boosters.adamax import *
from corerec.cr_boosters.optimizer import *
from corerec.cr_boosters.adadelta import *
from corerec.cr_boosters.adagrad import *
from corerec.cr_boosters.asgd import *
from corerec.cr_boosters.lbfgs import *
from corerec.cr_boosters.rmsprop import *
from corerec.cr_boosters.sgd import *
from corerec.cr_boosters.sparse_adam import *


#Promoted this script to engine.cr_utility.dataset
from corerec.datasets import GraphDataset


#FormatMaster is the plug for corerec preprocessing to detect dataset format and category
# from engine.format_master.ds_format_loader import *
# from engine.format_master.cr_formatMaster import *
# from engine.format_master.format_library import *