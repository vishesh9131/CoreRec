# Vendored `torch_nn` and `torch_utils`

CoreRec ships internal copies of PyTorch utility modules under:

- `corerec/torch_nn/`
- `corerec/torch_utils/`

## Why they exist

These trees were vendored to:

1. **Stabilize legacy engine code** that imported internal PyTorch paths before PyTorch 2.x API changes
2. **Support sandbox neural models** (`corerec/sandbox/collaborative_full/nn_base/`) without pinning an old PyTorch fork
3. **Isolate experimental code** from the production `corerec.engines` tier

## Production guidance

- **Production models** (`corerec/engines/`) should prefer **public PyTorch APIs** (`torch.nn`, `torch.optim`) directly
- Do **not** import `corerec.torch_nn` in new production code
- Sandbox / research code may continue using vendored modules until sandbox models graduate or are removed

## Maintenance policy

| Area | Policy |
|------|--------|
| `corerec/engines/` | No new dependencies on vendored torch trees |
| `corerec/sandbox/` | Allowed; migrate on graduation |
| Type checking | `mypy` ignores vendored paths (see `pyproject.toml`) |
| Trimming | Long-term goal: shrink vendored surface or gate behind optional extra |

## For framework users

If you only use **production models**, you can ignore `torch_nn/` entirely — it is not part of the public API.

```python
# ✅ Production import
from corerec.engines import DCN

# ⚠️ Internal / sandbox only
from corerec.torch_nn.modules.linear import Linear  # do not use in app code
```
