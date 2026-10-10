"""One id <-> code mapping for every model (#76).

Models each built their own dicts, some 0-based and some 1-based with 0 as
padding, under three names, with their own persistence. IdIndex is the one
place that factorizes ids, so stage 2 can move models onto it one at a time.
"""

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


class IdIndex:
    """Ids in first-appearance order, coded ``offset .. offset + len - 1``.

    ``offset=1`` leaves code 0 free for padding (sequence models).
    """

    def __init__(self, ids: Iterable[Any] = (), offset: int = 0):
        if not isinstance(offset, (int, np.integer)) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        self.offset = int(offset)
        self.ids: List[Any] = list(dict.fromkeys(ids))
        self._code: Dict[Any, int] = {x: k + offset for k, x in enumerate(self.ids)}

    @classmethod
    def fit(cls, values: Sequence[Any], offset: int = 0):
        """Index ``values`` and return ``(index, codes)``, codes as an int64 array."""
        codes, uniques = pd.factorize(pd.Series(list(values), dtype=object))
        if (codes < 0).any():
            raise ValueError("IDs must not be missing")
        return cls(uniques, offset), codes.astype(np.int64) + offset

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, x: Any) -> bool:
        return x in self._code

    def code(self, x: Any) -> int:
        return self._code[x]

    def codes(self, values: Iterable[Any]) -> np.ndarray:
        """Codes for known ids; raises KeyError naming the first unknown one."""
        return np.fromiter((self._code[x] for x in values), dtype=np.int64)

    def id(self, code: int) -> Any:
        if not isinstance(code, (int, np.integer)) or not self.offset <= code < self.offset + len(self):
            raise KeyError(code)
        return self.ids[code - self.offset]

    def as_dict(self) -> Dict[Any, int]:
        """The ``{id: code}`` dict models expose today as user_map / item_map."""
        return dict(self._code)

    def to_json(self) -> Dict[str, Any]:
        return {"ids": [x.item() if isinstance(x, np.generic) else x for x in self.ids],
                "offset": self.offset}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "IdIndex":
        return cls(data["ids"], data.get("offset", 0))
