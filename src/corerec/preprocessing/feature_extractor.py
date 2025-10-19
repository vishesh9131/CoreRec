import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class FeatureExtractor:
    """
    A class for extracting features from datasets, particularly text data.
    """

    def __init__(self, max_features=1000, stop_words='english'):
        """
        Initializes the FeatureExtractor with specified parameters.

        Parameters:
        - max_features (int): Maximum number of features to extract.
        - stop_words (str or list): Stop words to remove from the text.
        """
        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=self.stop_words)
        logging.info(f"FeatureExtractor initialized with max_features={self.max_features} and stop_words={self.stop_words}.")

    def extract_tfidf(self, df: pd.DataFrame, text_column: str) -> (pd.DataFrame, TfidfVectorizer):
        """
        Extracts TF-IDF features from a specified text column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the text data.
        - text_column (str): The name of the column containing text data.

        Returns:
        - pd.DataFrame: A DataFrame containing the TF-IDF features.
        - TfidfVectorizer: The fitted TF-IDF vectorizer.
        """
        if text_column not in df.columns:
            logging.error(f"Column '{text_column}' not found in DataFrame.")
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        logging.info(f"Extracting TF-IDF features from column: {text_column}")
        tfidf_matrix = self.vectorizer.fit_transform(df[text_column].fillna(''))
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        logging.info(f"Extracted {len(feature_names)} TF-IDF features.")
        
        return tfidf_df, self.vectorizer

    def extract_custom_features(self, df: pd.DataFrame, custom_func) -> pd.DataFrame:
        """
        Applies a custom feature extraction function to the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - custom_func (callable): A function that takes a DataFrame and returns a DataFrame of features.

        Returns:
        - pd.DataFrame: A DataFrame containing the custom features.
        """
        logging.info("Applying custom feature extraction function.")
        custom_features = custom_func(df)
        logging.info(f"Extracted custom features with shape: {custom_features.shape}")
        
        return custom_features