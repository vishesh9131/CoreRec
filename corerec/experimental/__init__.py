"""Untested code kept for reference (#79).

Nothing else in CoreRec imports these packages and no test covers them, so
they carry no stability or correctness promise:

- ``corerec.experimental.towers``: CNN/transformer/fusion towers (needs the
  ``transformers`` extra and torchvision). The towers the models use are
  ``corerec.core.towers``.
- ``corerec.experimental.integrations``: MLflow and Weights & Biases trackers.

They may change or be removed without a deprecation period.
"""
