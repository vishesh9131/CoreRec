from .PGM_uf_base import PGMUFBase as VE_PGM
from .BPRE_base import BayesianPersonalizedRankingExtensionsBase as VE_BPRE
from .Standard_VAE_base import StandardVAEBase as VE_STANDARD
from .Multinomial_VAE_base import MultinomialVAEBase as VE_MULTINOMIAL

__all__ = [
    "VE_PGM",
    "VE_BPRE",
    "VE_STANDARD",
    "VE_MULTINOMIAL",
]