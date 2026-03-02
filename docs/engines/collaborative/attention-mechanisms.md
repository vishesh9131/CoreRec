# Attention Mechanisms

Attention mechanisms allow the model to focus on the most relevant parts of a user's interaction history when making prediction.

## Overview

Useful for sequential data or when different items in history have varying importance.

## Available Models

### DIN (Deep Interest Network)
Adaptive interest modeling for click-through rate prediction.

::: corerec.engines.collaborative.attention_mechanism_base.din.DIN
    options:
      show_root_heading: true
      show_source: true

### SASRec
Self-Attentive Sequential Recommendation (also documented in Deep Learning section).

::: corerec.engines.sasrec.SASRec
    options:
      show_root_heading: true
      show_source: true
