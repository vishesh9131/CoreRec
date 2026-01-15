import unittest
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (
    REPO_ROOT /
    "corerec" /
    "engines" /
    "content_based" /
    "embedding_representation_learning")


def ensure_fake_package(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    m = ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m
    return m


def load_as(fqname: str, rel_file: str):
    file_path = BASE_DIR / rel_file
    spec = spec_from_file_location(fqname, str(file_path))
    mod = module_from_spec(spec)
    mod.__package__ = fqname.rsplit(".", 1)[0]
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sys.modules[fqname] = mod
    return mod


class TestPersonalizedEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fake_package("corerec", REPO_ROOT / "corerec")
        ensure_fake_package(
            "corerec.engines",
            REPO_ROOT /
            "corerec" /
            "engines")
        ensure_fake_package(
            "corerec.engines.content_based",
            REPO_ROOT / "corerec" / "engines" / "content_based",
        )
        ensure_fake_package(
            "corerec.engines.content_based.embedding_representation_learning",
            BASE_DIR)

        # Preload submodules explicitly to avoid importing the entire engines
        # package tree
        doc2vec_mod = load_as(
            "corerec.engines.content_based.embedding_representation_learning.doc2vec",
            "doc2vec.py",
        )
        load_as(
            "corerec.engines.content_based.embedding_representation_learning.word2vec",
            "word2vec.py",
        )
        cls.mod = load_as(
            "corerec.engines.content_based.embedding_representation_learning.personalized_embeddings",
            "personalized_embeddings.py",
        )

        # Stub WORD2VEC to avoid recursion from real implementation
        class _FakeW2V:
            def __init__(self, **kwargs):
                self.vector_size = int(kwargs.get("vector_size", 8))

            def train(self, sentences, epochs=1):
                return None

            def get_embedding(self, word):
                return [0.0] * self.vector_size

            def save_model(self, path):
                return None

            def load_model(self, path):
                return None

        cls.mod.WORD2VEC = _FakeW2V

        cls.PERSONALIZED_EMBEDDINGS = cls.mod.PERSONALIZED_EMBEDDINGS

    def test_doc2vec_flow(self):
        pe = self.PERSONALIZED_EMBEDDINGS(
            word2vec_params={"vector_size": 8},
            doc2vec_params={"vector_size": 8},
        )
        pe.train_doc2vec([["alpha", "beta"], ["gamma"]])
        vec = pe.get_doc_embedding(["alpha", "beta"])  # expected tokens
        self.assertGreater(len(vec), 0)
        self.assertLessEqual(len(vec), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
