# Neural Network Based Algorithms

These algorithms leverage neural network architectures to learn complex non-linear interactions between users and items, going beyond simple dot product interactions found in Matrix Factorization.

## Overview
Neural Collaborative Filtering (NCF) typically replaces the dot product with a multi-layer perceptron (MLP) to learn the user-item interaction function.

## Available Models

::: corerec.engines.collaborative.nn_base.neu_mf.NeuMF
    options:
      show_root_heading: true
      show_source: true

::: corerec.engines.collaborative.nn_base.gmf.GMF
    options:
      show_root_heading: true
      show_source: true

::: corerec.engines.collaborative.nn_base.mlp.MLP
    options:
      show_root_heading: true
      show_source: true
