# corerec/engines/contentFilterEngine/embedding_representation_learning/word2vec.py

from gensim.models import Word2Vec
from typing import List, Dict, Any

"""
Word2Vec Implementation for Word Embedding Generation

This module provides a wrapper class for Gensim's Word2Vec implementation, specifically
designed for generating word embeddings in recommendation systems. It focuses on
capturing semantic relationships between words in a continuous vector space.

Key Features:
    - Word-level embedding generation
    - Skip-gram and CBOW model support
    - Negative sampling optimization
    - Hierarchical softmax support
    - Thread-safe implementation

Example Usage:
    >>> word2vec = WORD2VEC(vector_size=100)
    >>> sentences = [['word1', 'word2'], ['word3', 'word4']]
    >>> word2vec.train(sentences)
    >>> embedding = word2vec.get_embedding('word1')

References:
    - Mikolov, et al. "Efficient estimation of word representations in vector space." 
      arXiv preprint arXiv:1301.3781 (2013).
"""

class WORD2VEC:
    """
    A Word2Vec model implementation for generating word embeddings.
    
    This class provides methods for training word embeddings and managing model
    persistence. It's particularly useful for recommendation systems that need
    to understand word-level semantics.

    Attributes:
        model (Word2Vec): The underlying Gensim Word2Vec model instance
        
    Methods:
        train: Trains the Word2Vec model on a corpus of sentences
        get_embedding: Retrieves the embedding vector for a specific word
        save_model: Persists the trained model to disk
        load_model: Loads a previously trained model from disk
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4):
        """
        Initialize a new Word2Vec model with specified parameters.

        Args:
            vector_size (int): Dimensionality of the word vectors. Higher dimensions can capture
                             more complex semantic relationships but require more data.
            window (int): Maximum distance between the current and predicted word within a sentence.
                         Larger windows consider broader context but may be noisier.
            min_count (int): Ignores all words with total frequency lower than this value.
                           Helps reduce noise from rare words.
            workers (int): Number of worker threads for training parallelization.
                         More workers can speed up training on multicore systems.

        Note:
            The model is not trained upon initialization. Call train() with your corpus
            to begin training.
        """
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def train(self, sentences: List[List[str]], epochs: int = 10):
        """
        Train the Word2Vec model on a corpus of sentences.

        Args:
            sentences (List[List[str]]): A list of tokenized sentences where each sentence
                                       is represented as a list of strings (tokens).
            epochs (int): Number of iterations over the corpus during training.
                         More epochs can improve quality but increase training time.

        Note:
            - Sentences should be preprocessed (tokenized, cleaned) before training
            - Training time scales with corpus size and vector_size
            - Progress can be monitored through Gensim's logging
        """
        self.model.build_vocab(sentences)
        self.model.train(sentences, total_examples=self.model.corpus_count, epochs=epochs)

    def get_embedding(self, word: str) -> List[float]:
        """
        Get the embedding vector for a given word.

        Parameters:
        - word (str): The word to retrieve the embedding for.

        Returns:
        - List[float]: The embedding vector.
        """
        if word in self.model.wv:
            return self.model.wv[word].tolist()
        else:
            return [0.0] * self.model.vector_size

    def save_model(self, path: str):
        """
        Save the trained Word2Vec model.

        Parameters:
        - path (str): File path to save the model.
        """
        self.model.save(path)

    def load_model(self, path: str):
        """
        Load a pre-trained Word2Vec model.

        Parameters:
        - path (str): File path of the saved model.
        """
        self.model = Word2Vec.load(path)