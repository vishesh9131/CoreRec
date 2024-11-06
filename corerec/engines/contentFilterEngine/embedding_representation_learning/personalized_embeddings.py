# corerec/engines/contentFilterEngine/embedding_representation_learning/personalized_embeddings.py

from typing import List, Dict, Any
from .word2vec import WORD2VEC
from .doc2vec import DOC2VEC

"""
Personalized Embeddings for Content-Based Recommendation

This module combines Word2Vec and Doc2Vec approaches to create personalized embeddings
for both individual words and complete documents in recommendation systems. It provides
a unified interface for training and managing both types of embeddings.

Key Features:
    - Unified Word2Vec and Doc2Vec management
    - Personalized document and word representations
    - Flexible parameter configuration
    - Model persistence capabilities
    - Thread-safe implementation

Example Usage:
    >>> embeddings = PERSONALIZED_EMBEDDINGS()
    >>> words = [['user', 'likes', 'movies'], ['user', 'watches', 'shows']]
    >>> docs = [['document', 'one', 'content'], ['document', 'two', 'content']]
    >>> embeddings.train_word2vec(words)
    >>> embeddings.train_doc2vec(docs)
    >>> word_vector = embeddings.get_word_embedding('user')
    >>> doc_vector = embeddings.get_doc_embedding(0)

References:
    - Mikolov, et al. "Efficient estimation of word representations in vector space." 2013.
    - Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." 2014.
"""

class PERSONALIZED_EMBEDDINGS:
    """
    A unified embedding manager combining Word2Vec and Doc2Vec capabilities.
    
    This class provides a comprehensive interface for training and managing both word
    and document embeddings, making it suitable for personalized recommendation systems
    that need to understand both word-level and document-level semantics.

    Attributes:
        word2vec (WORD2VEC): Instance of the Word2Vec model for word embeddings
        doc2vec (DOC2VEC): Instance of the Doc2Vec model for document embeddings
        
    Methods:
        train_word2vec: Trains the Word2Vec model on a corpus of sentences
        train_doc2vec: Trains the Doc2Vec model on a corpus of documents
        get_word_embedding: Retrieves word vectors
        get_doc_embedding: Retrieves document vectors
        save_models: Persists both models to disk
        load_models: Loads both models from disk
    """

    def __init__(self, word2vec_params: Dict[str, Any] = None, doc2vec_params: Dict[str, Any] = None):
        """
        Initialize both Word2Vec and Doc2Vec models with customizable parameters.

        Args:
            word2vec_params (Dict[str, Any], optional): Configuration parameters for Word2Vec model.
                                                       Includes vector_size, window, min_count, workers.
            doc2vec_params (Dict[str, Any], optional): Configuration parameters for Doc2Vec model.
                                                      Includes vector_size, window, min_count, workers, epochs.

        Note:
            If no parameters are provided, models will be initialized with default values.
            See individual model documentation for default parameter details.
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