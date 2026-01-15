import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_module(rel_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / rel_path
    spec = spec_from_file_location(mod_path.stem, str(mod_path))
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSimpleDoc2Vec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_module(
            "corerec/engines/content_based/embedding_representation_learning/doc2vec.py"
        )
        cls.SimpleDoc2Vec = cls.mod.SimpleDoc2Vec

    def test_train_and_get_embedding(self):
        model = self.SimpleDoc2Vec(vector_size=16)
        docs = [["this", "is", "doc1"], ["another", "doc"], ["more", "text"]]
        model.train(docs)
        vec = model.get_embedding(["this", "is", "doc1"])
        self.assertGreater(len(vec), 0)
        self.assertLessEqual(len(vec), 16)
        self.assertTrue(all(isinstance(float(x), float) for x in vec))


if __name__ == "__main__":
    unittest.main(verbosity=2)
