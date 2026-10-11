"""One adapter from any fit() input to one Interactions object (#78).

User input reaches models through up to six coercion layers today. This is
the single replacement: every accepted form (the triple, keywords, a
DataFrame, a corerec dataset) becomes the same Interactions, built on
IdIndex (#76). Stage 2 of #78 routes fit() through it one family at a time.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from corerec.api.exceptions import InvalidDataError
from corerec.api.id_index import IdIndex


@dataclass
class Interactions:
    """One row per event, ids coded through ``users`` / ``items``."""

    users: IdIndex
    items: IdIndex
    user_codes: np.ndarray        # int64 [n]
    item_codes: np.ndarray        # int64 [n]
    ratings: np.ndarray           # float32 [n]; ones when none were given
    timestamps: Optional[np.ndarray] = None  # float64 [n]

    def __len__(self) -> int:
        return len(self.user_codes)

    def triple(self):
        """(user_ids, item_ids, ratings) as lists, for fit() signatures not yet migrated."""
        return ([self.users.ids[c] for c in self.user_codes],
                [self.items.ids[c - self.items.offset] for c in self.item_codes],
                self.ratings.tolist())


def to_interactions(*args: Any, item_offset: int = 0, **kwargs: Any) -> Interactions:
    """Build Interactions from any form fit() accepts.

    ``to_interactions(users, items[, ratings])``, the same by keyword
    (``user_ids=``, ``item_ids=``, ``ratings=`` or ``interactions=``,
    ``timestamps=``), or one DataFrame / corerec dataset. Missing ratings
    mean 1.0. Raises InvalidDataError for mismatched lengths or non-finite
    ratings; ``item_offset=1`` leaves item code 0 for padding.
    """
    timestamps = kwargs.pop("timestamps", None)
    if len(args) == 1 and not kwargs:
        from corerec.api.dataset import as_interaction_frame

        frame = as_interaction_frame(args[0])
        if frame is None:
            raise InvalidDataError(f"can't read interactions from {type(args[0]).__name__}; "
                                   "pass (user_ids, item_ids[, ratings]) or a DataFrame "
                                   "with user_id and item_id columns")
        if timestamps is None and "timestamp" in frame.columns:
            timestamps = frame["timestamp"].to_numpy()
        users, items, ratings = frame["user_id"], frame["item_id"], frame["rating"]
    else:
        users = kwargs.pop("user_ids", args[0] if len(args) > 0 else None)
        items = kwargs.pop("item_ids", args[1] if len(args) > 1 else None)
        third = args[2] if len(args) > 2 else None
        ratings = kwargs.pop("ratings", kwargs.pop("interactions", third))
        if kwargs or len(args) > 3:
            raise TypeError(f"unexpected arguments: {sorted(kwargs) or args[3:]}")
        if users is None or items is None:
            raise TypeError("need user_ids and item_ids")
        if ratings is not None and (sp.issparse(ratings) or np.ndim(ratings) == 2):
            return _from_matrix(users, items, ratings, item_offset)

    users, items = list(users), list(items)
    if len(users) != len(items):
        raise InvalidDataError(f"{len(users)} user_ids but {len(items)} item_ids")
    r = np.ones(len(users), np.float32) if ratings is None else np.asarray(ratings, np.float64)
    if r.ndim != 1:
        raise InvalidDataError("ratings must be one-dimensional")
    if len(r) != len(users):
        raise InvalidDataError(f"{len(r)} ratings for {len(users)} interactions")
    bad = int((~np.isfinite(r)).sum())
    if bad:
        raise InvalidDataError(f"{bad} of {len(r)} ratings are NaN or infinite. "
                               "Drop or fill those rows before calling fit().")
    ts = None
    if timestamps is not None:
        ts = np.asarray(timestamps, np.float64)
        if ts.ndim != 1 or not np.isfinite(ts).all():
            raise InvalidDataError("timestamps must be a finite one-dimensional array")
        if len(ts) != len(users):
            raise InvalidDataError(f"{len(ts)} timestamps for {len(users)} interactions")

    r = r.astype(np.float32)
    if not np.isfinite(r).all():
        raise InvalidDataError("ratings exceed the finite float32 range")
    try:
        uix, ucodes = IdIndex.fit(users)
        iix, icodes = IdIndex.fit(items, offset=item_offset)
    except ValueError as exc:
        raise InvalidDataError(str(exc)) from exc
    return Interactions(uix, iix, ucodes, icodes, r, ts)


def _from_matrix(users: Any, items: Any, matrix: Any, item_offset: int) -> Interactions:
    """The matrix form: one user id per row, one item id per column, and every
    nonzero cell is an event. Ids with an empty row or column stay in the index."""
    users, items = list(users), list(items)
    m = sp.coo_matrix(matrix)
    if m.shape != (len(users), len(items)):
        raise InvalidDataError(f"a {m.shape[0]}x{m.shape[1]} interaction matrix needs {m.shape[0]} "
                               f"user_ids and {m.shape[1]} item_ids, got {len(users)} and {len(items)}")
    uix, iix = IdIndex(users), IdIndex(items, item_offset)
    if len(uix) != len(users) or len(iix) != len(items):
        raise InvalidDataError("with an interaction matrix, user_ids and item_ids name its rows "
                               "and columns, so each id may appear only once")
    keep = m.data != 0  # NaN != 0, so NaN cells are kept and rejected below
    r = np.asarray(m.data[keep], np.float64)
    bad = int((~np.isfinite(r)).sum())
    if bad:
        raise InvalidDataError(f"{bad} cells of the interaction matrix are NaN or infinite")
    return Interactions(uix, iix, m.row[keep].astype(np.int64),
                        m.col[keep].astype(np.int64) + item_offset, r.astype(np.float32))
