# corerec/engines/contentFilterEngine/embedding_representation_learning/word2vec.py

from gensim.models import Word2Vec
from typing import List, Dict, Any

class WORD2VEC:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4):
        """
        Initialize the Word2Vec model.

        Parameters:
        - vector_size (int): Dimensionality of the word vectors.
        - window (int): Maximum distance between the current and predicted word.
        - min_count (int): Ignores all words with total frequency lower than this.
        - workers (int): Number of worker threads to train the model.
        """
        self.model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    def train(self, sentences: List[List[str]], epochs: int = 10):
        """
        Train the Word2Vec model.

        Parameters:
        - sentences (List[List[str]]): A list of tokenized sentences.
        - epochs (int): Number of iterations (epochs) over the corpus.
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