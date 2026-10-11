# Constants

Default column names and similarity types used throughout CoreRec.

## Usage

```python
from corerec import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, SIM_COSINE

# SAR's default column names (the Microsoft Recommenders convention)
print(DEFAULT_USER_COL)   # "userID"
print(DEFAULT_ITEM_COL)   # "itemID"
print(DEFAULT_RATING_COL) # "rating"
print(SIM_COSINE)         # "cosine", one of SAR's similarity_type values
```

These are the defaults of `SAR(col_user=..., col_item=...)` and its similarity
types. The rest of CoreRec (`fit`, `evaluate`, `corerec train`, the cr_learn
frames) uses `user_id` / `item_id`; SAR also accepts that form, since
`SAR().fit(df)` reads a `user_id`/`item_id` frame too.

## API Reference

```{eval-rst}
.. automodule:: corerec.constants
   :members:
   :undoc-members:
```
