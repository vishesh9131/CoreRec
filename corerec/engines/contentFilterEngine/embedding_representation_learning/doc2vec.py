# corerec/engines/contentFilterEngine/embedding_representation_learning/doc2vec.py

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict, Any

class DOC2VEC:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4, epochs: int = 10):
        """
        Initialize the Doc2Vec model.

        Parameters:
        - vector_size (int): Dimensionality of the feature vectors.
        - window (int): Maximum distance between the current and predicted word.
        - min_count (int): Ignores all words with total frequency lower than this.
        - workers (int): Number of worker threads to train the model.
        - epochs (int): Number of training iterations.
        """
        self.model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

    def train(self, documents: List[List[str]]):
        """
        Train the Doc2Vec model.

        Parameters:
        - documents (List[List[str]]): A list of tokenized documents.
        """
        tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_embedding(self, doc_id: int) -> List[float]:
        """
        Get the embedding vector for a given document ID.

        Parameters:
        - doc_id (int): The document ID.

        Returns:
        - List[float]: The embedding vector.
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