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

Override them when initializing a model:

```python
from corerec.engines.collaborative import SAR

model = SAR(
    col_user='user_id',
    col_item='item_id',
    col_rating='rating',
    col_timestamp='ts',
)
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
