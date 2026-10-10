# Safe Model Bundles (`corerec_safe_v1`)

Registered production models default to safe bundles. Loading a legacy pickle or
full PyTorch checkpoint requires `allow_pickle=True` and emits a warning.
For a runnable save/load example, see {doc}`model_persistence`.

## Bundle layout

For the base path `production/dcn`, CoreRec writes:

| File | Contents |
|------|----------|
| `dcn.meta.json` | Constructor settings, fitted state, model class, component filenames |
| `dcn.<generation>.weights.pt` | PyTorch `state_dict`, loaded with `weights_only=True` |
| `dcn.<generation>.arrays.npz` | Numeric arrays, loaded with `allow_pickle=False` |

`<generation>` is a generated identifier. Only the components needed by the model
are written. Use the path passed to `save()` when calling `load()`; the metadata
records the component filenames.

New components are written before metadata is replaced atomically. A failed save
preserves the previous bundle. Successful saves remove the previous generated
components, and readers retry if a concurrent save replaced their generation.
Older bundles with fixed component filenames remain readable.

A `.pkl` or `.pt` suffix on the path does not select the format: `safe=True` is
still the default. Dotted base names, such as `model.v1`, remain distinct.

## ID maps and sparse matrices

User and item maps are stored as JSON pair lists so integer keys survive a round
trip. Classic CF, embedding CF, VAE, and the PyTorch wrapper store sparse matrices
as CSR `data`, `indices`, `indptr`, and `shape` arrays. Saving these matrices does
not allocate a dense user-by-item or item-by-item matrix.

## Detect and inspect a bundle

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from corerec.api.model_bundle import is_safe_bundle, load_bundle
from corerec.engines import ALS

model = ALS(factors=3, iterations=1).fit([1, 1, 2, 2], [10, 20, 20, 30])
with TemporaryDirectory() as directory:
    path = Path(directory) / "als"
    model.save(path)
    assert is_safe_bundle(path)
    bundle = load_bundle(path)
    assert bundle["metadata"]["model_class"] == "corerec.engines.matrix_factorization.ALS"
    assert "R__indptr" in bundle["arrays"]
```

## Legacy migration

Use `Model.load(legacy_path, allow_pickle=True)` only for a trusted file, then
`model.save(new_path)` to write a safe bundle. `safe=False` explicitly selects
legacy saving for models that support it.

| Model group | Safe components |
|-------------|-----------------|
| DCN, DeepFM, SASRec, TwoTower, HSTU | Metadata and weights; optional arrays |
| MultVAE, MultiDAE, `nn.Recommender` | Metadata, weights, and CSR arrays |
| ItemKNN, UserKNN, EASE, SLIM, ALS, Item2Vec | Metadata and numeric/CSR arrays |
| SAR, LightGCN, TFIDFRecommender | Metadata and arrays |

## Loading boundaries

Bundle component paths must remain inside the artifact directory. NumPy object
arrays are rejected, and PyTorch weights use its restricted loader.
`ModelLoader` selects registered CoreRec classes from metadata. Custom classes
must be passed explicitly; custom PyTorch modules require `module_cls=`.

## API reference

- `corerec.api.model_bundle.save_bundle` / `load_bundle`
- `corerec.api.torch_bundle.save_torch_production` / `load_torch_production`
- `corerec.api.bundle_helpers.pack_sparse_arrays` / `unpack_sparse_arrays`
- `corerec.api.bundle_helpers.save_map_state` / `load_map_state`
