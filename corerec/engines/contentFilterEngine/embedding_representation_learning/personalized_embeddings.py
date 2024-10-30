# corerec/engines/contentFilterEngine/embedding_representation_learning/personalized_embeddings.py

from typing import List, Dict, Any
from .word2vec import WORD2VEC
from .doc2vec import DOC2VEC

class PERSONALIZED_EMBEDDINGS:
    def __init__(self, word2vec_params: Dict[str, Any] = None, doc2vec_params: Dict[str, Any] = None):
        """
        Initialize the Personalized Embeddings recommender by setting up Word2Vec and Doc2Vec models.

        Parameters:
        - word2vec_params (dict, optional): Parameters for initializing Word2Vec.
        - doc2vec_params (dict, optional): Parameters for initializing Doc2Vec.
        """
        self.word2vec = WORD2VEC(**(word2vec_params if word2vec_params else {}))
        self.doc2vec = DOC2VEC(**(doc2vec_params if doc2vec_params else {}))

    def train_word2vec(self, sentences: List[List[str]], epochs: int = 10):
        """
        Train the Word2Vec model.

        Parameters:
        - sentences (List[List[str]]): A list of tokenized sentences.
        - epochs (int): Number of training iterations.
        """
        self.word2vec.train(sentences, epochs=epochs)

    def train_doc2vec(self, documents: List[List[str]]):
        """
        Train the Doc2Vec model.

        Parameters:
        - documents (List[List[str]]): A list of tokenized documents.
        """
        self.doc2vec.train(documents)

    def get_word_embedding(self, word: str) -> List[float]:
        """
        Get the embedding vector for a given word.

        Parameters:
        - word (str): The word to retrieve the embedding for.

        Returns:
        - List[float]: The embedding vector.
        """
        return self.word2vec.get_embedding(word)

    def get_doc_embedding(self, doc_id: int) -> List[float]:
        """
        Get the embedding vector for a given document ID.

        Parameters:
        - doc_id (int): The document ID.

        Returns:
        - List[float]: The embedding vector.
        """
        return self.doc2vec.get_embedding(doc_id)

    def save_models(self, word2vec_path: str, doc2vec_path: str):
        """
        Save both Word2Vec and Doc2Vec models.

        Parameters:
        - word2vec_path (str): File path to save the Word2Vec model.
        - doc2vec_path (str): File path to save the Doc2Vec model.
        """
        self.word2vec.save_model(word2vec_path)
        self.doc2vec.save_model(doc2vec_path)

    def load_models(self, word2vec_path: str, doc2vec_path: str):
        """
        Load pre-trained Word2Vec and Doc2Vec models.

        Parameters:
        - word2vec_path (str): File path of the saved Word2Vec model.
        - doc2vec_path (str): File path of the saved Doc2Vec model.
        """
        self.word2vec.load_model(word2vec_path)
        self.doc2vec.load_model(doc2vec_path)