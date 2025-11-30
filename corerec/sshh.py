import warnings
import logging

"""
Turn off tensorflow, keras, scipy, jax, pandas, matplotlib,
PIL, sklearn, nltk, gensim, torch, numpy warnings
"""

logging.getLogger("tensorflow").setLevel(logging.WARNING)
logging.getLogger("keras").setLevel(logging.WARNING)
logging.getLogger("scipy").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pandas").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("nltk").setLevel(logging.WARNING)
logging.getLogger("gensim").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)


# print("[corerec-dev/sshh.py] All warnings are sshhhh!!!") # Commented out for CLI
# by importing this file, all the warnings are turned off
# usage: from corerec.sshh import *
