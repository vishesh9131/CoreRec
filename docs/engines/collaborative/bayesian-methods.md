# Bayesian Methods

Bayesian methods incorporate uncertainty into the recommendation process, often performing better in cold-start or sparse data scenarios.

## Available Models

::: corerec.engines.collaborative.bayesian_method_base.bpr.BPR
    options:
      show_root_heading: true
      show_source: true

---

# Sequential Models

Sequential models treat recommendation as a sequence prediction problem, trying to predict the next item based on the ordered history.

## Available Models

::: corerec.engines.collaborative.sequential_model_base.caser.Caser
    options:
      show_root_heading: true
      show_source: true

::: corerec.engines.collaborative.sequential_model_base.gru4rec.GRU4Rec
    options:
      show_root_heading: true
      show_source: true
