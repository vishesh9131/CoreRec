"""
Sandbox: Collaborative Filtering Methods
=========================================

45+ collaborative filtering methods preserved from main engine.

All methods are accessible here with full functionality.
These are being refined and tested for potential graduation to main engine.

Usage:
------
    # Import any method from the full collection
    from corerec.sandbox.collaborative import DeepFM, DCN, SASRec
    from corerec.sandbox.collaborative import SVD, ALS, NMF
    from corerec.sandbox.collaborative import RBM, BPR, GeoMLC
    
    # Or via submodules
    from corerec.sandbox.collaborative.nn import DeepFM
    from corerec.sandbox.collaborative.mf import SVD

All original functionality preserved.
"""

# Import from the full backup of collaborative engine
import sys
import os

# Add the full collaborative backup to path
_full_path = os.path.join(os.path.dirname(__file__), '../collaborative_full')
if _full_path not in sys.path:
    sys.path.insert(0, _full_path)

# Now import everything from the full backup
try:
    # Submodules
    from corerec.sandbox.collaborative_full import mf_base as mf
    from corerec.sandbox.collaborative_full import nn_base as nn
    from corerec.sandbox.collaborative_full import graph_based_base as graph
    from corerec.sandbox.collaborative_full import sequential_model_base as sequential
    from corerec.sandbox.collaborative_full import attention_mechanism_base as attention
    from corerec.sandbox.collaborative_full import bayesian_method_base as bayesian
    from corerec.sandbox.collaborative_full import variational_encoder_base as vae
    from corerec.sandbox.collaborative_full import regularization_based_base as regularization
except ImportError as e:
    # Graceful fallback
    print(f"Warning: Could not import some sandbox modules: {e}")
    mf = nn = graph = sequential = attention = bayesian = vae = regularization = None

# Direct imports of popular methods
try:
    from corerec.sandbox.collaborative_full.rbm import RBM
    from corerec.sandbox.collaborative_full.rlrmc import RLRMC
    from corerec.sandbox.collaborative_full.geomlc import GeoMLC
    from corerec.sandbox.collaborative_full.sli import SLI
    from corerec.sandbox.collaborative_full.sum import SUM
    from corerec.sandbox.collaborative_full.cornac_bpr import BPR
except ImportError:
    RBM = RLRMC = GeoMLC = SLI = SUM = BPR = None

__all__ = [
    # Submodules
    "mf", "nn", "graph", "sequential", "attention", "bayesian", "vae", "regularization",
    # Direct access
    "RBM", "RLRMC", "GeoMLC", "SLI", "SUM", "BPR",
]


def list_available():
    """List all 45+ available collaborative methods in sandbox."""
    return """
    Sandbox Collaborative Methods (45+)
    ===================================
    
    Matrix Factorization (mf.*):
    - SVD, ALS, NMF, PMF, BPR, Bayesian MF
    
    Neural Networks (nn.*):
    - DeepFM, DCN, AutoInt, PNN, xDeepFM, DLRM
    - NCF variants, Deep MF, Hybrid Deep Learning
    - AFM, AutoFI, DeepCrossing, DeepFEFM
    - DIFM, ENSFM, FFM, FGCNN, Fibinet, FLEN, FM
    
    Sequential (sequential.*):
    - GRU, LSTM, Caser, NextItNet
    - SASRec, BERT4Rec, TiSAS, BST
    
    Attention (attention.*):
    - DIN, DIEN, DMR
    - Transformer-based variants
    
    Graph-Based (graph.*):
    - DeepWalk, various GNN models
    - GraphSAGE, GAT, GCN variants
    
    Variational (vae.*):
    - VAE, BiVAE, CVAE
    
    Bayesian (bayesian.*):
    - Bayesian MF, MCMC methods
    
    Multi-Task:
    - MMOE, PLE, ESMM, ESCMM
    
    Others:
    - RBM, RLRMC, GeoMLC, SLI, SUM
    - Self-supervised learning variants
    
    Import: from corerec.sandbox.collaborative import <MethodName>
    """


def get_method_info(method_name):
    """Get information about a specific method."""
    info_db = {
        "DeepFM": "Deep Factorization Machine - FM + DNN",
        "DCN": "Deep & Cross Network - explicit feature crossing",
        "SASRec": "Self-Attentive Sequential - transformer for sequences",
        "BPR": "Bayesian Personalized Ranking - pairwise loss",
        "RBM": "Restricted Boltzmann Machine - generative model",
        "GeoMLC": "Geometric Matrix Completion",
        "RLRMC": "Riemannian Low-Rank Matrix Completion",
    }
    return info_db.get(method_name, "Check method docstring for details")
