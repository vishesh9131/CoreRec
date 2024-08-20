"""
Engine
======
Maintainer: Vishesh


TL;DR
------
  1. If you're using this module, then surely you're in a testing phase for corerec.
  2. This module is not meant to be used by the end user.
  3. I'll fix the __init__; it's not yet written well. -Vishesh(Admin)

Provides
  1. Graph-based recommendation systems using neural network architectures.
  2. Various neural network modules and utilities.
  3. Training and prediction functions for graph data.

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `engine` has been imported as ``eng``::

  >>> import engine as eng

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(eng.train_model)
  ... # doctest: +SKIP

Available subpackages
---------------------
torch_nn
    Neural network modules and utilities.
cr_boosters
    Optimizers and boosters for training.
core_rec
    Core recommendation system components.
"""
