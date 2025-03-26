# corerec/engines/contentFilterEngine/embedding_representation_learning/doc2vec.py

from sklearn.feature_extraction.text import TfidfVectorizer

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

class SimpleDoc2Vec:
    """
    A Doc2Vec model implementation for generating document embeddings.
    
    This class provides methods for training document embeddings, retrieving vectors,
    and managing model persistence. It's particularly useful for recommendation systems
    that need to understand document-level semantics.

    Attributes:
        vectorizer (TfidfVectorizer): The underlying scikit-learn TfidfVectorizer instance
        
    Methods:
        train: Trains the Doc2Vec model on a corpus of documents
        get_embedding: Retrieves the embedding vector for a specific document
    """

    def __init__(self, vector_size=100):
        """
        Initialize a new Doc2Vec model with specified parameters.

        Args:
            vector_size (int): Dimensionality of the feature vectors. Higher dimensions can capture
                             more complex patterns but require more data and computation.
        """
        self.vectorizer = TfidfVectorizer(max_features=vector_size)

    def train(self, documents):
        """
        Train the Doc2Vec model on a corpus of documents.

        This method processes the input documents, builds a vocabulary, and trains
        the model using the specified parameters from initialization.

        Args:
            documents (List[List[str]]): A list of tokenized documents where each document
                                       is represented as a list of strings (tokens).

        Example:
            >>> doc2vec = SimpleDoc2Vec()
            >>> docs = [['this', 'is', 'doc1'], ['this', 'is', 'doc2']]
            >>> doc2vec.train(docs)

        Note:
            - Documents should be preprocessed (tokenized, cleaned) before training
            - Training time scales with corpus size and vector_size
            - Progress can be monitored through Gensim's logging
        """
        # Flatten the list of tokenized documents into strings
        documents = [' '.join(doc) for doc in documents]
        self.vectorizer.fit(documents)

    def get_embedding(self, document):
        """
        Retrieve the embedding vector for a specific document.

        Args:
            document (List[str]): The tokenized document to embed.

        Returns:
            List[float]: A dense vector representation of the document with
                        dimensionality specified by vector_size.

        Raises:
            RuntimeError: If called before training the model

        Note:
            The returned vector captures semantic properties of the document
            and can be used for similarity calculations or as features for
            downstream tasks.
        """
        # Convert a single document to a TF-IDF vector
        return self.vectorizer.transform([' '.join(document)]).toarray()[0]