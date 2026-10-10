# Data Preparation

## Interaction Data Format

CoreRec expects user-item interaction data as a Pandas DataFrame with these columns:

```python
import pandas as pd

interactions = pd.DataFrame({
    'userID': [1, 1, 2, 2, 3],
    'itemID': [101, 102, 101, 103, 102],
    'rating': [5.0, 3.0, 4.0, 2.0, 5.0],
    'timestamp': [1000, 1001, 1002, 1003, 1004],
})
```

## Default Column Names

CoreRec uses configurable column names with sensible defaults:

```python
from corerec import (
    DEFAULT_USER_COL,       # "userID"
    DEFAULT_ITEM_COL,       # "itemID"
    DEFAULT_RATING_COL,     # "rating"
    DEFAULT_TIMESTAMP_COL,  # "timestamp"
)
```

Override column names only when your DataFrame already uses those exact names:

```python
from corerec.engines.collaborative import SAR

# DataFrame columns must match col_* exactly (defaults: userID, itemID, rating)
interactions = pd.DataFrame({
    'userID': [1, 1, 2],
    'itemID': [101, 102, 101],
    'rating': [5.0, 3.0, 4.0],
})

model = SAR()  # defaults: col_user='userID', col_item='itemID'
model.fit(interactions)
```

If your raw CSV uses `user_id` / `item_id`, rename before `fit()`:

```python
df = df.rename(columns={'user_id': 'userID', 'movie_id': 'itemID'})
model = SAR()
model.fit(df)
```

## Supported Data Types

- **Explicit feedback**: ratings (1-5 stars, 1-10 scale)
- **Implicit feedback**: clicks, views, purchases (binary or count)
- **Sequential data**: ordered user interaction histories with timestamps

## Similarity Types

For collaborative filtering models like SAR:

```python
from corerec import (
    SIM_COSINE,            # Cosine similarity
    SIM_JACCARD,           # Jaccard similarity
    SIM_LIFT,              # Lift similarity
    SIM_COOCCURRENCE,      # Co-occurrence count
    SIM_MUTUAL_INFORMATION,# Mutual information
)
```


## Loading saved datasets and embeddings

`BaseDataset.save()` writes a legacy pickle file. Load it only when you trust
its source, with `BaseDataset.load(path, allow_pickle=True)`; this emits a
warning because pickle can execute Python. Failed saves preserve the previous
file.

For embedding tables, use numeric NumPy archives instead of pickle:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from corerec.embeddings.pretrained import PretrainedEmbeddings

with TemporaryDirectory() as directory:
    path = Path(directory) / "embeddings.npz"
    np.savez(path, embeddings=np.ones((2, 3)), ids=np.array(["01", "1"]))
    table = PretrainedEmbeddings.load(path)
    assert table.ids == ["01", "1"]
    assert table.dim == 3
```

Legacy `.pkl` embeddings require `PretrainedEmbeddings.load(path,
allow_pickle=True)` for a trusted file.

Generic `corerec.serialization` JSON/YAML loading reconstructs only classes
registered explicitly by the application. Import the intended class yourself
and call `SerializableRegistry.register(MyClass.__name__, MyClass)` before
loading. Generic pickle files require `load_from_file(path, allow_pickle=True)`;
`deserialize(path)` and `Serializable.load(path)` enforce the same rule.
