import numpy as np
import pytest

from corerec.embeddings import PretrainedEmbeddings


@pytest.mark.parametrize('vectors,ids', [([1, 2], ['a', 'b']),
                                         ([[1], [2]], ['a']),
                                         ([[1], [2]], ['a', 'a']),
                                         ([[np.nan]], ['a'])])
def test_invalid_embedding_tables_are_rejected(vectors, ids):
    with pytest.raises(ValueError):
        PretrainedEmbeddings(vectors, ids)


def test_empty_queries_and_zero_results():
    table = PretrainedEmbeddings([[1, 0], [0, 1]], ['a', 'b'])
    assert table.get_batch([]).shape == (0, 2)
    assert table.most_similar('a', top_k=0) == []
    with pytest.raises(ValueError):
        table.most_similar('a', top_k=-1)


@pytest.mark.parametrize('header', ['', '2 1\n'])
def test_one_dimensional_text_keeps_first_vector(tmp_path, header):
    path = tmp_path / 'vectors.txt'
    path.write_text(header + 'a 1.0\nb 2.0\n')
    table = PretrainedEmbeddings.load(path)
    assert table.ids == ['a', 'b']
    np.testing.assert_array_equal(table.get('a'), [1.0])
