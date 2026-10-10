import numpy as np
import pytest

from corerec.api.id_index import IdIndex
from corerec.api.interactions import to_interactions
from corerec.api.exceptions import InvalidDataError


@pytest.mark.parametrize("kwargs", [
    {"ratings": [[1], [2]]}, {"ratings": 1},
    {"ratings": [1e100, 2]}, {"timestamps": [[1], [2]]},
    {"timestamps": [np.nan, 2]},
])
def test_invalid_vectors_are_rejected(kwargs):
    with pytest.raises(InvalidDataError):
        to_interactions([1, 2], [10, 20], **kwargs)


def test_missing_ids_cannot_become_padding_or_negative_codes():
    with pytest.raises(InvalidDataError, match="missing"):
        to_interactions([1, None], [10, 20], item_offset=1)


@pytest.mark.parametrize("code", [0, -1, 3, 1.5])
def test_inverse_mapping_rejects_unknown_codes(code):
    index = IdIndex(["a", "b"], offset=1)
    with pytest.raises(KeyError):
        index.id(code)
