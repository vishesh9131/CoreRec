"""
Unified training dataset container for CoreRec production models.

Wraps the four supported ``fit()`` input shapes behind one type so pipelines
and training scripts can stay model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from corerec.api.exceptions import InvalidDataError


@dataclass
class RecommenderDataset:
    """
    Unified training data container.

    Supported modes (auto-detected via :meth:`infer_mode`):

    * **dataframe** — ``dataframe`` with user/item columns (SAR, NCF)
    * **triplet** — parallel ``user_ids``, ``item_ids``, ``ratings`` lists
    * **matrix** — ``interaction_matrix`` + ``user_ids`` + ``item_ids``
    * **content** — ``items`` + ``documents`` (TFIDF and content models)
    """

    user_ids: Optional[List[Any]] = None
    item_ids: Optional[List[Any]] = None
    ratings: Optional[List[float]] = None
    interaction_matrix: Optional[np.ndarray] = None
    dataframe: Optional[pd.DataFrame] = None
    items: Optional[List[Any]] = None
    documents: Optional[List[str]] = None
    user_col: str = "user_id"
    item_col: str = "item_id"
    rating_col: str = "rating"
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        **meta: Any,
    ) -> "RecommenderDataset":
        return cls(
            dataframe=df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            meta=meta,
        )

    @classmethod
    def from_triplet(
        cls,
        user_ids: List[Any],
        item_ids: List[Any],
        ratings: Optional[List[float]] = None,
        **meta: Any,
    ) -> "RecommenderDataset":
        return cls(user_ids=list(user_ids), item_ids=list(item_ids), ratings=ratings, meta=meta)

    @classmethod
    def from_matrix(
        cls,
        user_ids: List[Any],
        item_ids: List[Any],
        interaction_matrix: np.ndarray,
        **meta: Any,
    ) -> "RecommenderDataset":
        return cls(
            user_ids=list(user_ids),
            item_ids=list(item_ids),
            interaction_matrix=np.asarray(interaction_matrix),
            meta=meta,
        )

    @classmethod
    def from_content(
        cls,
        items: List[Any],
        documents: List[str],
        **meta: Any,
    ) -> "RecommenderDataset":
        return cls(items=list(items), documents=list(documents), meta=meta)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def infer_mode(self) -> str:
        if self.documents is not None and self.items is not None:
            return "content"
        if self.dataframe is not None:
            return "dataframe"
        if self.interaction_matrix is not None:
            return "matrix"
        if self.user_ids is not None and self.item_ids is not None:
            return "triplet"
        raise InvalidDataError(
            "RecommenderDataset is incomplete. Provide one of:\n"
            "  - dataframe\n"
            "  - user_ids + item_ids + ratings (triplet)\n"
            "  - user_ids + item_ids + interaction_matrix (matrix)\n"
            "  - items + documents (content)"
        )

    def as_sar_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with userID/itemID columns for SAR."""
        if self.dataframe is not None:
            df = self.dataframe.copy()
            rename = {}
            if self.user_col in df.columns and self.user_col != "userID":
                rename[self.user_col] = "userID"
            if self.item_col in df.columns and self.item_col != "itemID":
                rename[self.item_col] = "itemID"
            if rename:
                df = df.rename(columns=rename)
            return df
        if self.user_ids is None or self.item_ids is None:
            raise InvalidDataError("Need dataframe or triplet data for SAR.")
        data = {"userID": self.user_ids, "itemID": self.item_ids}
        if self.ratings is not None:
            data["rating"] = self.ratings
        return pd.DataFrame(data)

    def as_ncf_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy()
        if self.user_ids is None or self.item_ids is None:
            raise InvalidDataError("Need dataframe or triplet data for NCF.")
        data = {self.user_col: self.user_ids, self.item_col: self.item_ids}
        if self.ratings is not None:
            data[self.rating_col] = self.ratings
        return pd.DataFrame(data)

    def as_triplet(self) -> tuple:
        self.infer_mode()  # validate
        if self.user_ids is None or self.item_ids is None:
            raise InvalidDataError("Triplet mode requires user_ids and item_ids.")
        return self.user_ids, self.item_ids, self.ratings

    def as_matrix(self) -> tuple:
        if self.interaction_matrix is None:
            raise InvalidDataError("Matrix mode requires interaction_matrix.")
        if self.user_ids is None or self.item_ids is None:
            raise InvalidDataError("Matrix mode requires user_ids and item_ids.")
        return self.user_ids, self.item_ids, self.interaction_matrix

    def as_content(self) -> tuple:
        if self.items is None or self.documents is None:
            raise InvalidDataError("Content mode requires items and documents.")
        return self.items, self.documents


def is_recommender_dataset(obj: Any) -> bool:
    return isinstance(obj, RecommenderDataset)


def coerce_dataset(first_arg: Any) -> Optional[RecommenderDataset]:
    """Return RecommenderDataset if *first_arg* is one, else None."""
    if isinstance(first_arg, RecommenderDataset):
        return first_arg
    if isinstance(first_arg, pd.DataFrame):
        return RecommenderDataset.from_dataframe(first_arg)
    return None
