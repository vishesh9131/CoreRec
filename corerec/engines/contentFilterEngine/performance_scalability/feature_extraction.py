# feature_extraction implementation
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
from typing import List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class FeatureExtraction:
    def __init__(self, max_features=5000):
        """
        Initializes the FeatureExtraction with a TF-IDF vectorizer.

        Parameters:
        - max_features (int): The maximum number of features (vocabulary size).
        """
        self.max_features = max_features
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',  # Use built-in stop words
            tokenizer=self.tokenize
        )
        logger.info(f"FeatureExtraction initialized with max_features={self.max_features}.")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes and lemmatizes the input text.

        Parameters:
        - text (str): The text to tokenize.

        Returns:
        - list: A list of processed tokens.
        """
        tokens = nltk.word_tokenize(text.lower())
        lemmatized = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha()
        ]
        logger.debug(f"Tokenized text: {lemmatized}")
        return lemmatized

    def fit_transform(self, documents: List[str]):
        """
        Fits the TF-IDF vectorizer on the documents and transforms them into feature vectors.

        Parameters:
        - documents (list of str): The list of documents to process.

        Returns:
        - sparse matrix: The TF-IDF feature matrix.
        """
        logger.info("Fitting and transforming documents into TF-IDF features.")
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents: List[str]) -> Any:
        """
        Transforms the documents into TF-IDF feature vectors using the already fitted vectorizer.

        Parameters:
        - documents (list of str): The list of documents to transform.

        Returns:
        - sparse matrix: The TF-IDF feature matrix.
        """
        logger.info("Transforming documents into LSA latent space.")
        tfidf_matrix = self.vectorizer.transform(documents)  # Use transform, not fit
        return self.lsa_model.transform(tfidf_matrix)

    def get_feature_names(self) -> List[str]:
        """
        Retrieves the feature names (vocabulary) from the vectorizer.

        Returns:
        - list: A list of feature names.
        """
        return self.vectorizer.get_feature_names_out()
