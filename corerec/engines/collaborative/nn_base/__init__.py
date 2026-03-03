"""
Neural-network-based collaborative filtering algorithms.

Imports are wrapped in try/except so that a single missing dependency
does not prevent the entire sub-package from loading.
"""

import importlib as _il
import logging as _log

_logger = _log.getLogger(__name__)

_IMPORTS = {
    "NN_SASREC": (".sasrec_base", "SASRecBase"),
    "NN_RNN_SEQUENTIAL_RECOMMENDATION": (".rnn_sequential_recommendation_base", "RNN_sequential_recommendation_base"),
    "NN_SELF_SUPERVISED_LEARNING_CF": (".self_supervised_learning_cf_base", "SelfSupervisedLearningCFBase"),
    "NN_XDEEPFM": (".xdeepfm_base", "XDeepFMBase"),
    "NN_NCF": (".ncf_base", "NCF_base"),
    "NN_AUTOENCODER_CF": (".autoencoder_cf_base", "AutoencoderCFBase"),
    "NN_HYBRID_DEEP_LEARNING": (".hybrid_deep_learning_base", "HybridDeepLearningBase"),
    "NN_NEXTITNET": (".nextitnet_base", "NextItNetBase"),
    "NN_SSEPT": (".ssept_base", "SSEPTBase"),
    "NN_BERT4REC": (".Bert4Rec_base", "Bert4Rec_base"),
    "NN_GNN": (".gnn_ufilter_base", "GNN_ufilter_base"),
    "NN_GAN": (".gan_ufilter_base", "GAN_ufilter_base"),
    "NN_TRANSFORMER": (".transformer_ufilter_base", "TransformerUFilterBase"),
    "NN_VARIATIONAL_AUTOENCODER": (".variational_autoencoder_ufilter_base", "VariationalAutoencoderUFilterBase"),
    "NN_SLIREC": (".slirec_base", "SLiRecBase"),
    "NN_SSR": (".SSR_base", "SSR_base"),
    "NN_DEEP_MF": (".deep_mf_base", "DeepMF_base"),
    "NN_BVAE": (".bivae_base", "BiVAE_base"),
    "NN_NEURAL_MF": (".neural_mf_base", "NeuralMFBase"),
    "NN_FM": (".FM_base", "FM_base"),
    "NN_DEEPFM": (".DeepFM_base", "DeepFM_base"),
    "NN_PNN": (".PNN_base", "PNN_base"),
    "NN_AFM": (".AFM_base", "AFM_base"),
    "NN_BST": (".BST_base", "BST_base"),
    "NN_NFM": (".NFM_base", "NFM_base"),
    "NN_ENSFM": (".ENSFM_base", "ENSFM_base"),
    "NN_DEEPREC": (".DeepRec_base", "DeepRec_base"),
    "NN_DEEPFEFM": (".DeepFEFM_base", "DeepFEFM_base"),
    "NN_GNN_BASE": (".GNN_base", "GNN_base"),
    "NN_PLE": (".PLE_base", "PLE_base"),
    "NN_GRU": (".gru_ufilter_base", "GRU_ufilter_base"),
    "NN_FFM": (".FFM_base", "FFM_base"),
    "NN_FGCNN": (".FGCNN_base", "FGCNN_base"),
    "NN_NEURAL_CF": (".neural_ufilter_base", "NeuralCollaborativeFilteringBase"),
    "NN_RALM": (".RALM_base", "RALM_base"),
    "NN_AUTOFI": (".AutoFI_base", "AutoFI_base"),
    "NN_ESMM": (".ESMM_base", "ESMM_base"),
    "NN_ESCMM": (".ESCMM_base", "ESCMM_base"),
    "NN_DMR": (".DMR_base", "DMR_base"),
    "NN_CASER": (".caser_base", "Caser_base"),
    "NN_LISTWISE": (".ListWise_base", "ListWise_base"),
    "NN_DIFM": (".DIFM_base", "DIFM_base"),
    "NN_DEEPCROSSING": (".DeepCrossing_base", "DeepCrossing_base"),
    "NN_DCN": (".DCN_base", "DCNModel"),
    "NN_MMOE": (".MMOE_base", "MMOE_base"),
    "NN_FIBINET": (".Fibinet_base", "Fibinet_base"),
    "NN_DLRM": (".DLRM_base", "DLRM_base"),
    "NN_AUTOINTER": (".AutoInt_base", "AutoInt_base"),
    "NN_MAML": (".MAML_base", "MAML_base"),
    "NN_GATENET": (".GateNet_base", "GateNet_base"),
    "NN_DIEN": (".DIEN_base", "DIEN_base"),
    "NN_FLEN": (".FLEN_base", "FLEN_base"),
    "NN_TISAS": (".TiSAS_base", "TiSASBase"),
}

def __getattr__(name):
    if name in _IMPORTS:
        mod_path, cls_name = _IMPORTS[name]
        try:
            mod = _il.import_module(mod_path, __name__)
            cls = getattr(mod, cls_name)
            globals()[name] = cls
            return cls
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            _logger.debug("Optional import %s failed: %s", name, exc)
            globals()[name] = None
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_IMPORTS.keys())
