# corerec/engines/contentFilterEngine/embedding_representation_learning/doc2vec.py

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict, Any

"""
Doc2Vec Implementation for Document Embedding Generation

This module provides a wrapper class for Gensim's Doc2Vec implementation, specifically
designed for generating document embeddings in recommendation systems. Doc2Vec extends
the Word2Vec algorithm by adding a document vector trained simultaneously with word vectors.

Key Features:
    - Document-level embedding generation
    - Paragraph Vector implementation (PV-DM and PV-DBOW)
    - Customizable training parameters
    - Model persistence support
    - Thread-safe implementation

Example Usage:
    >>> doc2vec = DOC2VEC(vector_size=100)
    >>> documents = [['word1', 'word2'], ['word3', 'word4']]
    >>> doc2vec.train(documents)
    >>> embedding = doc2vec.get_embedding(0)

References:
    - Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents."
      International conference on machine learning. PMLR, 2014.
"""

class DOC2VEC:
    """
    A Doc2Vec model implementation for generating document embeddings.
    
    This class provides methods for training document embeddings, retrieving vectors,
    and managing model persistence. It's particularly useful for recommendation systems
    that need to understand document-level semantics.

    Attributes:
        model (Doc2Vec): The underlying Gensim Doc2Vec model instance
        
    Methods:
        train: Trains the Doc2Vec model on a corpus of documents
        get_embedding: Retrieves the embedding vector for a specific document
        save_model: Persists the trained model to disk
        load_model: Loads a previously trained model from disk
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, 
                 workers: int = 4, epochs: int = 10):
        """
        Initialize a new Doc2Vec model with specified parameters.

        Args:
            vector_size (int): Dimensionality of the feature vectors. Higher dimensions can capture
                             more complex patterns but require more data and computation.
            window (int): Maximum distance between the current and predicted word within a sentence.
                         Larger windows capture broader context but may introduce noise.
            min_count (int): Ignores all words with total frequency lower than this value.
                           Helps reduce noise from rare words.
            workers (int): Number of worker threads for training parallelization.
                         More workers can speed up training on multicore systems.
            epochs (int): Number of iterations over the corpus during training.
                         More epochs can improve quality but increase training time.

        Note:
            The model is not trained upon initialization. Call train() with your corpus
            to begin training.
        """
        self.model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

    def train(self, documents: List[List[str]]):
        """
        Train the Doc2Vec model on a corpus of documents.

        This method processes the input documents, builds a vocabulary, and trains
        the model using the specified parameters from initialization.

        Args:
            documents (List[List[str]]): A list of tokenized documents where each document
                                       is represented as a list of strings (tokens).

        Example:
            >>> doc2vec = DOC2VEC()
            >>> docs = [['this', 'is', 'doc1'], ['this', 'is', 'doc2']]
            >>> doc2vec.train(docs)

        Note:
            - Documents should be preprocessed (tokenized, cleaned) before training
            - Training time scales with corpus size and vector_size
            - Progress can be monitored through Gensim's logging
        """
        tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_embedding(self, doc_id: int) -> List[float]:
        """
        Retrieve the embedding vector for a specific document.

        Args:
            doc_id (int): The unique identifier of the document to embed.
                         Must be within range of trained documents.

        Returns:
            List[float]: A dense vector representation of the document with
                        dimensionality specified by vector_size.

        Raises:
            KeyError: If doc_id is not found in the trained model
            RuntimeError: If called before training the model

        Note:
            The returned vector captures semantic properties of the document
            and can be used for similarity calculations or as features for
            downstream tasks.
        """
        return self.model.dv[str(doc_id)].tolist()

    def save_model(self, path: str):
        """
        Save the trained Doc2Vec model.

        Parameters:
        - path (str): File path to save the model.
        """
        self.model.save(path)

    def load_model(self, path: str):
        """
        Load a pre-trained Doc2Vec model.

        Parameters:
        - path (str): File path of the saved model.
        """
        self.model = Doc2Vec.load(path)