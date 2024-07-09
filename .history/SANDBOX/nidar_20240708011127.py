import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Your code here