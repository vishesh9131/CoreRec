# Safe Model Bundles (`corerec_safe_v1`)

Production models default to **safe bundles** instead of pickle or full PyTorch checkpoints.

```python
model.save("/artifacts/dcn")          # safe=True (default)
loaded = DCN.load("/artifacts/dcn")   # auto-detects format
```

## Bundle layout

Given base path `/artifacts/dcn`, CoreRec writes:

| File | Contents |
|------|----------|
| `dcn.meta.json` | Config, JSON-safe state (maps as pairs), format version |
| `dcn.weights.pt` | Torch `state_dict` (loaded with `weights_only=True`) |
| `dcn.arrays.npz` | Numeric arrays only (loaded with `allow_pickle=False`) |

Torch-only models use `.meta.json` + `.weights.pt`.  
Numpy/sparse models (SAR, FAST, LightGCN, TFIDF) use `.meta.json` + `.arrays.npz`.

Metadata header:

```json
{
  "format": "corerec_safe_v1",
  "corerec_save_version": "1.0",
  "model_class": "corerec.engines.dcn.DCN",
  "config": { "...": "constructor kwargs" },
  "state": { "...": "fitted state, maps as *_pairs lists" }
}
```

## ID maps and JSON

User/item IDs are stored as **pair lists** (`user_map_pairs`) so integer IDs survive JSON round-trip. Do not store raw dicts with integer keys in `state` — JSON stringifies keys and breaks `predict(0, item)`.

## Legacy migration

### Detect format

```python
from corerec.api.model_bundle import is_safe_bundle

is_safe_bundle("/artifacts/dcn")       # True → safe bundle
is_safe_bundle("/artifacts/legacy.pt") # False → legacy
```

### Re-save to safe format

```python
model = SAR.load("legacy.pkl")   # legacy still loads
model.save("production/sar", safe=True)
```

### Opt-in legacy save (discouraged)

```python
model.save("legacy.pkl", safe=False)  # pickle or torch checkpoint
```

**Security:** Only load legacy artifacts from trusted sources. Pickle and `torch.load(weights_only=False)` can execute arbitrary code.

### Format by model

| Model group | Safe files | Legacy fallback |
|-------------|------------|-----------------|
| DCN, DeepFM, GNNRec, MIND, NASRec, TwoTower, BERT4Rec, SASRec, NCF | `.meta.json` + `.weights.pt` | `.pt` checkpoint |
| SAR, FAST, FASTRecommender, LightGCN, TFIDF | `.meta.json` + `.arrays.npz` | `.pkl` / `.npy` |

## Verification after load

Always verify inference, not just `is_fitted`:

```python
score_before = model.predict(user_id, item_id)
loaded = Model.load(path)
score_after = loaded.predict(user_id, item_id)
assert abs(score_before - score_after) < 1e-2
```

## API reference

- `corerec.api.model_bundle.save_bundle` / `load_bundle`
- `corerec.api.torch_bundle.save_torch_production` / `load_torch_production`
- `corerec.api.bundle_helpers.save_map_state` / `load_map_state`

See also: {doc}`model_persistence`.
