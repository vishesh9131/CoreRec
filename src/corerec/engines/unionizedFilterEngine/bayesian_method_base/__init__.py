from .Bayesian_mf_base import BayesianMFBase as BAYESIAN_MF
from .Bayesian_Personalized_Ranking_Extensions_base import BayesianPersonalizedRankingBase as BAYESIAN_PERSONALIZED_RANKING
from .PGM_uf import PGMUFBase as PGM_UF
# from .bivae import BiVAE as BAYESIAN_BiVAE

from .bpr_base import BPRBase as BAYESIAN_BPR
from .bprmf_base import BPRMFBase as BAYESIAN_BPRMF
from .multinomial_vae import MultinomialVAE as BAYESIAN_vae
from .PGM_uf import PGMUFBase as BAYESIAN_PGMUF
from .vmf_base import VMFBase as BAYESIAN_VMF

__all__ = [
    "BAYESIAN_MF",
    "BAYESIAN_PERSONALIZED_RANKING",
    "PGM_UF",
    "BAYESIAN_BiVAE",
]
