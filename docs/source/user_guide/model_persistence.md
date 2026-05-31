# Model Persistence

## Saving models (production default)

All 14 production models support safe bundles (`corerec_safe_v1`):

```python
model.save("/artifacts/my_model")   # safe=True by default
```

This writes JSON metadata plus separate weight/array files — no pickle in the default path.

## Loading models

```python
from corerec.engines import DCN

loaded = DCN.load("/artifacts/my_model")  # auto-detects safe vs legacy
recommendations = loaded.recommend(user_id=1, top_k=10)
```

## Safe bundle specification

See {doc}`safe_bundle_persistence` for file layout, ID map encoding, and migration from legacy `.pkl` / `.pt` artifacts.

## Legacy formats

```python
model.save("legacy.pkl", safe=False)  # pickle/torch — untrusted files are unsafe
```

Migrate existing deployments:

```python
legacy = SAR.load("old/sar.pkl")
legacy.save("prod/sar", safe=True)
```

## Verify round-trip quality

```python
uid, iid = 42, 1001
assert abs(model.predict(uid, iid) - loaded.predict(uid, iid)) < 1e-2
recs = loaded.recommend(uid, top_k=10)
```

## Model information

```python
info = model.get_model_info()
print(f"Model: {info['name']}")
print(f"Fitted: {info['is_fitted']}")
```

## Checking model state

```python
if model.is_fitted:
    recs = model.recommend(user_id=1, top_k=10)

if model.knows_user(42):
    score = model.predict(user_id=42, item_id=100)
```
