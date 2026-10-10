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

    users, items = list(users), list(items)
    if len(users) != len(items):
        raise InvalidDataError(f"{len(users)} user_ids but {len(items)} item_ids")
    r = np.ones(len(users), np.float32) if ratings is None else np.asarray(ratings, np.float64)
    if len(r) != len(users):
        raise InvalidDataError(f"{len(r)} ratings for {len(users)} interactions")
    bad = int((~np.isfinite(r)).sum())
    if bad:
        raise InvalidDataError(f"{bad} of {len(r)} ratings are NaN or infinite. "
                               "Drop or fill those rows before calling fit().")
    ts = None
    if timestamps is not None:
        ts = np.asarray(timestamps, np.float64)
        if len(ts) != len(users):
            raise InvalidDataError(f"{len(ts)} timestamps for {len(users)} interactions")

    uix, ucodes = IdIndex.fit(users)
    iix, icodes = IdIndex.fit(items, offset=item_offset)
    return Interactions(uix, iix, ucodes, icodes, r.astype(np.float32), ts)
