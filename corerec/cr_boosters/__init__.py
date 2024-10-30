# """
# :mod:`torch.optim` is a package implementing various optimization algorithms.

# Most commonly used methods are already supported, and the interface is general
# enough, so that more sophisticated ones can also be easily integrated in the
# future.
# """

# from torch.optim import lr_scheduler, swa_utils
# from torch.optim.adadelta import Adadelta
# from torch.optim.adagrad import Adagrad
# from torch.optim.adam import Adam
# from torch.optim.adamax import Adamax
# from torch.optim.adamw import AdamW
# from torch.optim.asgd import ASGD
# from torch.optim.lbfgs import LBFGS
# from torch.optim.nadam import NAdam
# from torch.optim.optimizer import Optimizer
# from torch.optim.radam import RAdam
# from torch.optim.rmsprop import RMSprop
# from torch.optim.rprop import Rprop
# from torch.optim.sgd import SGD
# from torch.optim.sparse_adam import SparseAdam

# del Adadelta  # type: ignore[name-defined] # noqa: F821
# del Adagrad  # type: ignore[name-defined] # noqa: F821
# del Adam  # type: ignore[name-defined] # noqa: F821
# del AdamW  # type: ignore[name-defined] # noqa: F821
# del SparseAdam  # type: ignore[name-defined] # noqa: F821
# del Adamax  # type: ignore[name-defined] # noqa: F821
# del ASGD  # type: ignore[name-defined] # noqa: F821
# del SGD  # type: ignore[name-defined] # noqa: F821
# del RAdam  # type: ignore[name-defined] # noqa: F821
# del Rprop  # type: ignore[name-defined] # noqa: F821
# del RMSprop  # type: ignore[name-defined] # noqa: F821
# del Optimizer  # type: ignore[name-defined] # noqa: F821
# del NAdam  # type: ignore[name-defined] # noqa: F821
# del LBFGS  # type: ignore[name-defined] # noqa: F821