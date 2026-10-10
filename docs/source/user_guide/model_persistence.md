# Model Persistence

Registered production models save safe bundles by default: JSON metadata, numeric NumPy arrays, and tensor bytes stored in NumPy archives without pickle.

## Save and load a model

This example trains, saves, loads, and checks prediction parity:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from corerec.engines import ItemKNN
from corerec.serving import ModelLoader

model = ItemKNN().fit([1, 1, 2, 2], [10, 20, 20, 30])
with TemporaryDirectory() as directory:
    path = Path(directory) / "itemknn"
    model.save(path)
    loaded = ModelLoader().load(path)
    assert loaded.predict(1, 10) == model.predict(1, 10)
    assert loaded.recommend(1, top_k=1) == [30]
```

You can also load a bundle with its model class, for example `ItemKNN.load(path)`.
See {doc}`safe_bundle_persistence` for the layout and sparse matrix representation.

## Legacy formats

Legacy pickle and full PyTorch checkpoints require explicit trust. Loading them can execute Python code; use the option only for files you trust.

To migrate a model you created yourself:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from corerec.engines import DCN

model = DCN(epochs=1, embedding_dim=4, deep_layers=[4], device="cpu")
model.fit([1, 1, 2, 2], [10, 20, 20, 30], [1., 1., 1., 1.])
with TemporaryDirectory() as directory:
    legacy_path = Path(directory) / "legacy.pt"
    model.save(legacy_path, safe=False)
    legacy = DCN.load(legacy_path, allow_pickle=True)
    safe_path = Path(directory) / "production"
    legacy.save(safe_path)
    assert DCN.load(safe_path).predict(1, 10) == model.predict(1, 10)

```

`ModelLoader.load()` and the CSV artifact loader also accept `allow_pickle=True`.
For a trusted legacy artifact directory, the equivalent CLI options are
`corerec serve ARTIFACT --allow-pickle` and `corerec retrain ARTIFACT --allow-pickle`.
Newly trained artifacts use safe bundles and need no trust option.

## Custom PyTorch modules

For `corerec.nn.Recommender`, built-in modules load automatically. Supply a custom
module class explicitly with `Recommender.load(path, module_cls=MyModule)`.
Custom loss functions also require an explicit `loss=my_loss` argument on load;
the artifact records that a custom loss is needed and never substitutes BPR.
Artifacts created before this check may already have lost their custom loss;
pass the original `loss=` explicitly when loading those artifacts.
An artifact's metadata cannot authorize importing arbitrary Python modules.

## Model information

```python
info = model.get_model_info()
assert info["is_fitted"]
print(info["model_type"], info["num_users"], info["num_items"])
```
