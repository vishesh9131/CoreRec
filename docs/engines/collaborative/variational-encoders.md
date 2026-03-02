# Variational Encoders

Variational Autoencoders (VAEs) for collaborative filtering extend standard autoencoders by learning a probability distribution for user preferences, providing robust regularization.

## Overview

Effective for implicit feedback and typically trained by maximizing the Evidence Lower Bound (ELBO).

## Available Models

### Multi-VAE
Multinomial Variational Autoencoder.

::: corerec.engines.collaborative.variational_encoder_base.multivae.MultiVAE
    options:
      show_root_heading: true
      show_source: true
