import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from corerec.engines.contentFilterEngine.miscellaneous_techniques import (
    MIS_FEATURE_SELECTION,
    MIS_NOISE_HANDLING,
    MIS_COLD_START
)
from sklearn.preprocessing import MultiLabelBinarizer

class MovieRecommender:
    def __init__(self):
        self.feature_selector = MIS_FEATURE_SELECTION(k=50, method='chi2')
        self.noise_handler = MIS_NOISE_HANDLING(method='isolation_forest', contamination=0.1)
        self.cold_start_handler = MIS_COLD_START(method='hybrid', n_neighbors=10)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.mlb = MultiLabelBinarizer()
        self.movies_df = None
        self.features = None
        self.processed_features = None

    def load_data(self, filepath):
        # Load the movies dataset
        try:
            self.movies_df = pd.read_csv(
                filepath,
                sep='::', 
                engine='python',
                header=None,
                names=['movie_id', 'title', 'genres'],
                encoding='utf-8'
            )
        except UnicodeDecodeError:
            self.movies_df = pd.read_csv(
                filepath,
                sep='::', 
                engine='python',
                header=None,
                names=['movie_id', 'title', 'genres'],
                encoding='latin1'
            )
        
        # Create feature matrix from movie titles and genres
        title_features = self.vectorizer.fit_transform(self.movies_df['title']).toarray()
        genre_features = self._encode_genres(self.movies_df['genres'])
        self.features = np.hstack([title_features, genre_features])
        
        return self.movies_df

    def _encode_genres(self, genres_series):
        # Convert genres string to one-hot encoding
        genres_split = genres_series.str.split('|')
        self.mlb.fit(genres_split)
        genre_matrix = self.mlb.transform(genres_split)
        return genre_matrix

    def preprocess_data(self):
        # Handle noisy data
        clean_features, clean_indices = self.noise_handler.fit_transform(self.features)
        self.movies_df = self.movies_df.iloc[clean_indices].reset_index(drop=True)
        
        # Select important features
        self.processed_features = self.feature_selector.fit_transform(
            clean_features, 
            self.movies_df['genres'].str.split('|').apply(len)
        )
        
        # Prepare data for cold start handling
        # In a real-world scenario, you'd use actual user-item interactions
        # Here, we simulate with random interactions for demonstration
        interaction_matrix = np.random.randint(0, 2, size=(100, len(self.movies_df)))
        self.cold_start_handler.fit(
            self.processed_features,
            interaction_matrix=interaction_matrix
        )
        
        return self.processed_features

    def get_recommendations(self, user_profile=None, n_recommendations=5):
        if user_profile is None:
            # Cold start scenario without user profile
            recommended_indices = self.cold_start_handler.recommend_for_new_user()
        else:
            # Cold start scenario with user profile
            # Transform user profile to match the feature pipeline
            user_vector = self._process_user_profile(user_profile)
            recommended_indices = self.cold_start_handler.recommend_for_new_user(user_vector)
        
        recommendations = self.movies_df.iloc[recommended_indices]
        return recommendations[['title', 'genres']]

    def _process_user_profile(self, user_profile):
        # Step 1: Vectorize the user profile
        user_tfidf = self.vectorizer.transform([user_profile]).toarray()
        
        # Step 2: Encode genres as zeros (since user_profile may not have genres)
        # Assuming genres are not part of the user_profile input
        genre_features = np.zeros((1, len(self.mlb.classes_)))
        
        # Step 3: Combine title and genre features
        user_features = np.hstack([user_tfidf, genre_features])
        
        # Step 4: Handle noisy data (if applicable)
        # Since it's a single user profile, we'll skip noise handling
        
        # Step 5: Feature selection
        user_selected = self.feature_selector.transform(user_features)
        
        # Step 6: Scale the features
        user_scaled = self.cold_start_handler.scaler.transform(user_selected)
        
        return user_scaled[0]

    def get_feature_importance(self):
        return pd.Series(
            self.feature_selector.get_feature_importance(),
            index=list(self.vectorizer.get_feature_names_out()) + list(self.mlb.classes_)
        ).sort_values(ascending=False)

def main():
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load and preprocess data
    print("Loading data...")
    movies_df = recommender.load_data('src/SANDBOX/dataset/ml-1m/movies.dat')
    print(f"Loaded {len(movies_df)} movies")
    
    print("\nPreprocessing data...")
    processed_features = recommender.preprocess_data()
    print(f"Features shape after preprocessing: {processed_features.shape}")
    
    # Get recommendations for new user
    print("\nGenerating recommendations for new user...")
    new_user_recs = recommender.get_recommendations()
    print("\nTop recommendations for new user:")
    print(new_user_recs)
    
    # Get recommendations for user with profile
    print("\nGenerating recommendations for user with profile...")
    user_profile = "Action adventure sci-fi movies"
    profile_recs = recommender.get_recommendations(user_profile=user_profile)
    print(f"\nTop recommendations for profile '{user_profile}':")
    print(profile_recs)
    
    # Show top important features
    print("\nTop important features:")
    print(recommender.get_feature_importance().head(10))

if __name__ == "__main__":
    main()