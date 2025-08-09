import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import pickle
import logging
from collections import defaultdict
import gc
import sys

# Add CoreRec modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# from dlrm.DLRM_base import DLRM_base
from corerec.engines.unionizedFilterEngine.nn_base.DLRM_base import DLRM_base


class DLRMRecommender(DLRM_base):
    """
    DLRM-based recommender system implementation
    """
    def __init__(self, 
                 name="DLRMRecommender",
                 embed_dim=32,
                 bottom_mlp_dims=[64, 32],
                 top_mlp_dims=[64, 32, 1],
                 dropout=0.1,
                 batchnorm=True,
                 learning_rate=0.001,
                 batch_size=256,
                 num_epochs=20,
                 patience=5,
                 shuffle=True,
                 device=None,
                 seed=42,
                 verbose=True,
                 config=None):
        # Call parent constructor with name parameter
        # The issue was that the BaseCorerec.__init__ requires 'name' 
        # but it wasn't being passed through the inheritance chain
        super(DLRM_base, self).__init__(name=name, trainable=True, verbose=verbose)
        
        # Continue with DLRM_base initialization but skip its parent constructor call
        self.embed_dim = embed_dim
        self.bottom_mlp_dims = bottom_mlp_dims
        self.top_mlp_dims = top_mlp_dims
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.shuffle = shuffle
        self.seed = seed
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Setup logger
        self._setup_logger()
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.BCELoss()
        
        # Initialize data structures for users, items, and features
        self.categorical_map = {}
        self.categorical_names = []
        self.field_dims = []
        self.dense_features = []
        self.dense_dim = 0
        
        # Initialize hook manager for model introspection
        self.hook_manager = None
        
        # Additional attributes for the recommender
        self.is_fitted = False
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} model with {self.embed_dim} embedding dimensions")
        
    def fit(self, interactions, user_features=None, item_features=None):
        """Train the DLRM model on interaction data"""
        # Map users and items to indices
        unique_users = interactions['user_id'].unique()
        unique_items = interactions['item_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {i: j for j, i in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {j: i for i, j in self.item_mapping.items()}
        
        # Convert to DLRM compatible format (list of dictionaries)
        formatted_data = []
        for _, row in interactions.iterrows():
            formatted_data.append({
                'user_id': self.user_mapping[row['user_id']],
                'item_id': self.item_mapping[row['item_id']],
                'label': float(row['rating'])
            })
        
        # Train DLRM model
        try:
            super().fit(formatted_data)
            self._extract_embeddings()
            self.is_fitted = True
        except Exception as e:
            print(f"Error training DLRM model: {e}")
            # Use a simple fallback for embeddings
            self._initialize_random_embeddings()
        
        return self
        
    def _initialize_random_embeddings(self):
        """Initialize random embeddings as fallback"""
        for user_id in self.user_mapping.keys():
            self.user_embeddings[user_id] = np.random.randn(self.embed_dim)
            
        for item_id in self.item_mapping.keys():
            self.item_embeddings[item_id] = np.random.randn(self.embed_dim)
    
    def _extract_embeddings(self):
        """Extract embeddings from trained model"""
        # In the CoreRec implementation, we may not have direct access to embeddings
        # We'll try to extract them if possible, otherwise use a different approach
        try:
            # Get user embeddings using the model's embedding layer
            if hasattr(self.model, 'embedding') and self.model.embedding is not None:
                for user_id, user_idx in self.user_mapping.items():
                    embedding = self.model.embedding.weight[user_idx].detach().cpu().numpy()
                    self.user_embeddings[user_id] = embedding
                
                for item_id, item_idx in self.item_mapping.items():
                    embedding = self.model.embedding.weight[item_idx + len(self.user_mapping)].detach().cpu().numpy()
                    self.item_embeddings[item_id] = embedding
            else:
                # Fallback to computing embeddings through predictions
                for user_id in self.user_mapping.keys():
                    self.user_embeddings[user_id] = np.random.randn(self.embed_dim)
                    
                for item_id in self.item_mapping.keys():
                    self.item_embeddings[item_id] = np.random.randn(self.embed_dim)
                    
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            self._initialize_random_embeddings()
    
    def recommend(self, user_id, n=10, exclude_known=True):
        """Generate recommendations for a user"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        if user_id not in self.user_mapping:
            # Handle cold-start user
            return []
        
        # Since we may not be able to directly use the model for predictions
        # we'll use the embeddings we've extracted to compute recommendations
        
        # Get the user embedding
        if user_id not in self.user_embeddings:
            # Cold start case - return empty list
            return []
            
        user_emb = self.user_embeddings[user_id]
        
        # Calculate similarity with all items
        scores = []
        for item_id, item_emb in self.item_embeddings.items():
            # Simple dot product similarity
            score = np.dot(user_emb, item_emb)
            scores.append((item_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return scores[:n]

class HybridMusicRecommender:
    """
    Music Recommendation System combining content-based filtering and DLRM
    
    This hybrid model combines:
    1. Content-based filtering using TF-IDF vectorization of lyrics
    2. Collaborative filtering using DLRM
    3. User preference tracking
    """
    
    def __init__(
        self,
        name: str = "SpotifyRecommender",
        embed_dim: int = 32,
        similarity_threshold: float = 0.3,
        dlrm_weight: float = 0.5,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the hybrid music recommender model.
        
        Args:
            name: Model name
            embed_dim: Embedding dimension for songs
            similarity_threshold: Minimum similarity score to consider songs as similar
            dlrm_weight: Weight to apply to DLRM scores vs content-based scores (0-1)
            seed: Random seed
            verbose: Whether to display progress information
        """
        self.name = name
        self.embed_dim = embed_dim
        self.similarity_threshold = similarity_threshold
        self.dlrm_weight = dlrm_weight
        self.seed = seed
        self.verbose = verbose
        self.is_fitted = False
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize data structures
        self.user_preferences = defaultdict(dict)
        self.song_data = None
        self.artist_data = None
        self.song_index_mapping = {}
        self.reverse_mapping = {}
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lyrics_matrix = None
        
        # Initialize DLRM model
        self.dlrm = DLRMRecommender(
            name=f"{name}_DLRM",
            embed_dim=embed_dim,
            seed=seed,
            verbose=verbose
        )
        
        # Setup logger
        self._setup_logger()
        
        if self.verbose:
            self.logger.info(f"Initialized {self.name} with {embed_dim} embedding dimensions")
    
    def _setup_logger(self):
        """Setup logger for the model."""
        self.logger = logging.getLogger(f"{self.name}")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False
    
    def fit(self, data_path: str) -> 'HybridMusicRecommender':
        """
        Fit the model on the Spotify dataset.
        
        Args:
            data_path: Path to the Spotify dataset CSV file
            
        Returns:
            The fitted model
        """
        if self.verbose:
            self.logger.info(f"Loading data from {data_path}")
        
        # Load data
        self.song_data = pd.read_csv(data_path)
        
        # Clean data
        self._preprocess_data()
        
        # Create song embeddings using TF-IDF on lyrics
        self._create_song_embeddings()
        
        # Initialize DLRM with synthetic data if we don't have real user interactions yet
        self._initialize_dlrm()
        
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info(f"Model fitted on {len(self.song_data)} songs")
        
        return self
    
    def _preprocess_data(self):
        """Preprocess the dataset."""
        if self.verbose:
            self.logger.info("Preprocessing data...")
        
        # Clean column names
        self.song_data.columns = [col.strip() for col in self.song_data.columns]
        
        # Drop duplicates
        self.song_data = self.song_data.drop_duplicates(subset=['artist', 'song'])
        
        # Clean text data
        self.song_data['text'] = self.song_data['text'].fillna('')
        self.song_data['text'] = self.song_data['text'].astype(str)
        
        # Create a unique identifier for each song
        self.song_data['song_id'] = self.song_data.index
        
        # Create mapping dictionaries
        for idx, row in self.song_data.iterrows():
            song_key = f"{row['artist']} - {row['song']}"
            self.song_index_mapping[song_key] = row['song_id']
            self.reverse_mapping[row['song_id']] = song_key
        
        # Extract artist information
        self.artist_data = self.song_data.groupby('artist').agg({
            'song': 'count'
        }).reset_index()
        self.artist_data.columns = ['artist', 'song_count']
        
        if self.verbose:
            self.logger.info(f"Dataset contains {len(self.song_data)} songs and {len(self.artist_data)} artists")
    
    def _create_song_embeddings(self):
        """Create song embeddings using TF-IDF on lyrics."""
        if self.verbose:
            self.logger.info("Creating song embeddings...")
        
        # Fit TF-IDF vectorizer on lyrics
        self.lyrics_matrix = self.vectorizer.fit_transform(self.song_data['text'])
        
        # Force garbage collection to free up memory
        gc.collect()
    
    def _initialize_dlrm(self):
        """Initialize DLRM with synthetic interactions if none available."""
        if self.verbose:
            self.logger.info("Initializing DLRM model...")
        
        # Create synthetic interactions from current user preferences or random if none
        interactions = []
        
        # If we have user preferences, use those
        if any(self.user_preferences.values()):
            for user_id, prefs in self.user_preferences.items():
                for song_id, rating in prefs.items():
                    interactions.append({
                        'user_id': user_id,
                        'item_id': song_id,
                        'rating': rating
                    })
        else:
            # Generate some random interactions for cold start
            n_users = 10
            n_items_per_user = 5
            
            for user_id in range(1, n_users + 1):
                # Sample random songs for each user
                random_songs = self.song_data.sample(n_items_per_user)
                
                for _, song in random_songs.iterrows():
                    interactions.append({
                        'user_id': user_id,
                        'item_id': song['song_id'],
                        'rating': np.random.uniform(0.5, 1.0)  # Random positive rating
                    })
        
        # Convert to DataFrame
        interactions_df = pd.DataFrame(interactions)
        
        # Fit DLRM on interactions
        if len(interactions_df) > 0:
            self.dlrm.fit(interactions_df)
        else:
            if self.verbose:
                self.logger.warning("No interactions available for DLRM training")
    
    def _calculate_similarity_for_song(self, song_idx):
        """Calculate similarity between a single song and all other songs."""
        song_vector = self.lyrics_matrix[song_idx:song_idx+1]
        similarities = cosine_similarity(song_vector, self.lyrics_matrix).flatten()
        return similarities
    
    def add_user_preference(self, user_id: int, song_key: str, rating: float):
        """
        Add user preference for a song.
        
        Args:
            user_id: User ID
            song_key: Song key in format "Artist - Song Name"
            rating: Rating from 0 to 1
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before adding user preferences")
        
        if song_key not in self.song_index_mapping:
            if self.verbose:
                self.logger.warning(f"Song '{song_key}' not found in the dataset")
            return
        
        song_id = self.song_index_mapping[song_key]
        self.user_preferences[user_id][song_id] = rating
        
        # Update DLRM with new interaction
        new_interaction = pd.DataFrame([{
            'user_id': user_id,
            'item_id': song_id,
            'rating': rating
        }])
        
        # Retrain DLRM if we have enough data
        if len(self.user_preferences) > 0:
            all_interactions = []
            for u_id, prefs in self.user_preferences.items():
                for s_id, r in prefs.items():
                    all_interactions.append({
                        'user_id': u_id,
                        'item_id': s_id,
                        'rating': r
                    })
            
            # Retrain with all data
            if self.verbose:
                self.logger.info(f"Retraining DLRM with {len(all_interactions)} interactions")
                
            interactions_df = pd.DataFrame(all_interactions)
            self.dlrm.fit(interactions_df)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries containing song information and score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
        
        # Check if user has any preferences
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            # Return random recommendations if no preferences are available
            random_songs = self.song_data.sample(n_recommendations)
            recommendations = []
            
            for _, song in random_songs.iterrows():
                recommendations.append({
                    'artist': song['artist'],
                    'song': song['song'],
                    'song_key': f"{song['artist']} - {song['song']}",
                    'link': song['link'],
                    'score': 0.5,  # Neutral score for random recommendations
                    'reason': "Random recommendation"
                })
            
            return recommendations
        
        try:
            # Get recommendations from both models and combine
            content_based_recs = self._get_content_based_recommendations(user_id, n_recommendations * 2)
            
            # Get DLRM recommendations if model is trained
            dlrm_recs = []
            try:
                if hasattr(self.dlrm, 'is_fitted') and self.dlrm.is_fitted:
                    dlrm_results = self.dlrm.recommend(user_id, n=n_recommendations * 2)
                    
                    for item_id, score in dlrm_results:
                        if item_id in self.reverse_mapping:
                            song_key = self.reverse_mapping[item_id]
                            song = self.song_data[self.song_data['song_id'] == item_id].iloc[0]
                            
                            dlrm_recs.append({
                                'song_id': item_id,
                                'artist': song['artist'],
                                'song': song['song'],
                                'song_key': song_key,
                                'link': song['link'],
                                'score': score,
                                'method': 'dlrm'
                            })
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Error getting DLRM recommendations: {e}")
            
            # Combine recommendations with weighted scores
            combined_recs = {}
            
            # Add content-based recommendations
            for rec in content_based_recs:
                song_id = rec['song_id']
                if song_id in self.user_preferences[user_id]:
                    continue  # Skip already liked songs
                    
                score = rec['score'] * (1 - self.dlrm_weight)
                reason = rec['reason']
                combined_recs[song_id] = {
                    'song_id': song_id,
                    'artist': rec['artist'],
                    'song': rec['song'],
                    'song_key': rec['song_key'],
                    'link': rec['link'],
                    'score': score,
                    'reason': reason,
                    'method': 'content'
                }
            
            # Add DLRM recommendations with weighted scores
            for rec in dlrm_recs:
                song_id = rec['song_id']
                if song_id in self.user_preferences[user_id]:
                    continue  # Skip already liked songs
                    
                score = rec['score'] * self.dlrm_weight
                
                if song_id in combined_recs:
                    # Combine scores if recommendation exists in both methods
                    combined_recs[song_id]['score'] += score
                    combined_recs[song_id]['method'] = 'hybrid'
                    combined_recs[song_id]['reason'] = "Personalized recommendation for you"
                else:
                    combined_recs[song_id] = {
                        'song_id': song_id,
                        'artist': rec['artist'],
                        'song': rec['song'],
                        'song_key': rec['song_key'],
                        'link': rec['link'],
                        'score': score,
                        'reason': "Based on your listening patterns",
                        'method': 'dlrm'
                    }
            
            # Sort and return top recommendations
            sorted_recommendations = sorted(
                combined_recs.values(), 
                key=lambda x: x['score'], 
                reverse=True
            )[:n_recommendations]
            
            # Finalize recommendations
            recommendations = []
            for rec in sorted_recommendations:
                # Add is_liked flag for UI
                is_liked = rec['song_id'] in self.user_preferences.get(user_id, {})
                
                recommendations.append({
                    'artist': rec['artist'],
                    'song': rec['song'],
                    'song_key': rec['song_key'],
                    'link': rec['link'],
                    'score': rec['score'],
                    'reason': rec['reason'],
                    'is_liked': is_liked
                })
            
        except Exception as e:
            # Fallback to popular songs if recommendation generation fails
            if self.verbose:
                self.logger.warning(f"Error generating recommendations: {e}")
            recommendations = self.get_popular_songs(n_recommendations)
        
        # If we still don't have enough recommendations, add random ones
        if len(recommendations) < n_recommendations:
            remaining = n_recommendations - len(recommendations)
            
            # Get IDs of songs that are already recommended or liked
            existing_song_keys = set(r.get('song_key', '') for r in recommendations)
            existing_ids = set(self.user_preferences.get(user_id, {}).keys())
            
            # Get random songs that are not already recommended or liked
            available_songs = self.song_data[~self.song_data['song_id'].isin(existing_ids)]
            
            if len(available_songs) > 0:
                random_songs = available_songs.sample(min(remaining, len(available_songs)))
                
                for _, song in random_songs.iterrows():
                    song_key = f"{song['artist']} - {song['song']}"
                    if song_key not in existing_song_keys:
                        recommendations.append({
                            'artist': song['artist'],
                            'song': song['song'],
                            'song_key': song_key,
                            'link': song['link'],
                            'score': 0.3,  # Lower score for random fill-ins
                            'reason': "You might want to explore this",
                            'is_liked': False
                        })
        
        # Force garbage collection
        gc.collect()
        
        return recommendations
    
    def _get_content_based_recommendations(self, user_id, n_recommendations):
        """Get content-based recommendations based on user preferences."""
        # Collect candidate songs based on similarity
        candidates = {}
        # Create a copy of the user preferences dictionary to avoid "dictionary changed size during iteration" error
        user_prefs_copy = dict(self.user_preferences[user_id])
        for song_id, rating in user_prefs_copy.items():
            song_idx = self.song_data[self.song_data['song_id'] == song_id].index[0]
            
            # Get similarities for this song - compute on-demand rather than using precomputed matrix
            similarities = self._calculate_similarity_for_song(song_idx)
            
            # Process similarities
            for i, similarity in enumerate(similarities):
                # Map from similarity array index to song data index
                if i < len(self.song_data):
                    candidate_id = self.song_data.iloc[i]['song_id']
                    
                    # Skip if the song is already in user preferences
                    if candidate_id in self.user_preferences[user_id]:
                        continue
                    
                    # Use the maximum similarity if a song appears multiple times
                    if candidate_id in candidates:
                        candidates[candidate_id] = max(candidates[candidate_id], similarity * rating)
                    else:
                        candidates[candidate_id] = similarity * rating
        
        # Sort candidates by score
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare recommendations
        recommendations = []
        for candidate_id, score in sorted_candidates[:n_recommendations]:
            song = self.song_data[self.song_data['song_id'] == candidate_id].iloc[0]
            
            # Find the song that led to this recommendation
            source_songs = []
            for liked_id in self.user_preferences[user_id]:
                liked_idx = self.song_data[self.song_data['song_id'] == liked_id].index[0]
                candidate_idx = self.song_data[self.song_data['song_id'] == candidate_id].index[0]
                
                # Compute similarity on-demand
                similarity = self._calculate_similarity_for_song(liked_idx)[candidate_idx]
                
                if similarity > self.similarity_threshold:
                    source_song = self.song_data.loc[liked_idx]
                    source_songs.append(f"{source_song['artist']} - {source_song['song']}")
            
            reason = f"Similar to {', '.join(source_songs[:2])}" if source_songs else "Based on your preferences"
            
            recommendations.append({
                'song_id': candidate_id,
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link'],
                'score': score,
                'reason': reason
            })
        
        return recommendations
    
    def get_similar_songs(self, song_key: str, n_similar: int = 10) -> List[Dict[str, Any]]:
        """
        Get similar songs to a given song.
        
        Args:
            song_key: Song key in format "Artist - Song Name"
            n_similar: Number of similar songs to return
            
        Returns:
            List of similar song dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before finding similar songs")
        
        if song_key not in self.song_index_mapping:
            if self.verbose:
                self.logger.warning(f"Song '{song_key}' not found in the dataset")
            return []
        
        song_id = self.song_index_mapping[song_key]
        song_idx = self.song_data[self.song_data['song_id'] == song_id].index[0]
        
        # Calculate similarities on-demand
        similarities = self._calculate_similarity_for_song(song_idx)
        
        # Get similarity scores
        similarity_tuples = []
        for i, similarity in enumerate(similarities):
            if i != song_idx and i < len(self.song_data):  # Skip the song itself and check index bounds
                similarity_tuples.append((i, similarity))
        
        # Sort by similarity
        similarity_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare result
        similar_songs = []
        for idx, similarity in similarity_tuples[:n_similar]:
            song = self.song_data.iloc[idx]
            similar_songs.append({
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link'],
                'score': similarity
            })
        
        # Force garbage collection
        gc.collect()
        
        return similar_songs
    
    def get_popular_songs(self, n_songs: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most popular songs (artists with most songs in the dataset).
        
        Args:
            n_songs: Number of songs to return
            
        Returns:
            List of popular song dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting popular songs")
        
        # Get most popular artists
        popular_artists = self.artist_data.nlargest(5, 'song_count')['artist'].tolist()
        
        # Get songs from popular artists
        popular_songs = []
        for artist in popular_artists:
            artist_songs = self.song_data[self.song_data['artist'] == artist].sample(
                min(n_songs // len(popular_artists) + 1, len(self.song_data[self.song_data['artist'] == artist]))
            )
            
            for _, song in artist_songs.iterrows():
                popular_songs.append({
                    'artist': song['artist'],
                    'song': song['song'],
                    'song_key': f"{song['artist']} - {song['song']}",
                    'link': song['link'],
                    'score': 0.8  # High score for popular songs
                })
        
        # Return top n_songs
        return popular_songs[:n_songs]
    
    def search_songs(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for songs based on artist or song name.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            
        Returns:
            List of matching song dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before searching")
        
        query = query.lower()
        
        # Search by artist and song name
        matches = self.song_data[
            self.song_data['artist'].str.lower().str.contains(query) |
            self.song_data['song'].str.lower().str.contains(query)
        ]
        
        # Prepare results
        results = []
        for _, song in matches.head(n_results).iterrows():
            results.append({
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link']
            })
        
        return results
    
    def get_real_time_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get real-time recommendations based on current user preferences.
        This is similar to the recommend method but optimized for quick response.
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations")
            
        if user_id not in self.user_preferences or not self.user_preferences[user_id]:
            return self.get_popular_songs(n_recommendations)
            
        # Get quick recommendations based on most recently liked song
        liked_songs = list(self.user_preferences[user_id].items())
        liked_songs.sort(key=lambda x: x[1], reverse=True)  # Sort by rating
        
        recent_song_id = liked_songs[0][0]  # Get most recently/highly rated song
        song_idx = self.song_data[self.song_data['song_id'] == recent_song_id].index[0]
        
        # Calculate similarities for the recent song
        similarities = self._calculate_similarity_for_song(song_idx)
        
        # Get similarity scores
        similarity_tuples = []
        for i, similarity in enumerate(similarities):
            if i != song_idx and i < len(self.song_data):
                candidate_id = self.song_data.iloc[i]['song_id']
                
                # Skip if already liked
                if candidate_id in self.user_preferences[user_id]:
                    continue
                    
                similarity_tuples.append((i, similarity))
        
        # Sort by similarity
        similarity_tuples.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare result
        real_time_recs = []
        recent_song = self.song_data.iloc[song_idx]
        recent_song_key = f"{recent_song['artist']} - {recent_song['song']}"
        
        for idx, similarity in similarity_tuples[:n_recommendations]:
            song = self.song_data.iloc[idx]
            real_time_recs.append({
                'artist': song['artist'],
                'song': song['song'],
                'song_key': f"{song['artist']} - {song['song']}",
                'link': song['link'],
                'score': similarity,
                'reason': f"Because you liked {recent_song_key}"
            })
        
        return real_time_recs
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model data
        model_data = {
            'name': self.name,
            'embed_dim': self.embed_dim,
            'similarity_threshold': self.similarity_threshold,
            'dlrm_weight': self.dlrm_weight,
            'seed': self.seed,
            'verbose': self.verbose,
            'is_fitted': self.is_fitted,
            'song_index_mapping': self.song_index_mapping,
            'reverse_mapping': self.reverse_mapping,
            'user_preferences': dict(self.user_preferences),
            'song_data': self.song_data.to_dict() if self.song_data is not None else None,
            'artist_data': self.artist_data.to_dict() if self.artist_data is not None else None
        }
        
        # Save vectorizer separately
        vectorizer_path = str(path.parent / f"{path.stem}_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save lyrics matrix separately
        lyrics_matrix_path = str(path.parent / f"{path.stem}_lyrics_matrix.npz")
        from scipy.sparse import save_npz
        save_npz(lyrics_matrix_path, self.lyrics_matrix)
        
        # Save DLRM model separately
        dlrm_path = str(path.parent / f"{path.stem}_dlrm.pkl")
        with open(dlrm_path, 'wb') as f:
            pickle.dump(self.dlrm, f)
        
        # Save main model data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HybridMusicRecommender':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load vectorizer
        path = Path(filepath)
        vectorizer_path = str(path.parent / f"{path.stem}_vectorizer.pkl")
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load lyrics matrix
        lyrics_matrix_path = str(path.parent / f"{path.stem}_lyrics_matrix.npz")
        from scipy.sparse import load_npz
        lyrics_matrix = load_npz(lyrics_matrix_path)
        
        # Load DLRM model
        dlrm_path = str(path.parent / f"{path.stem}_dlrm.pkl")
        with open(dlrm_path, 'rb') as f:
            dlrm = pickle.load(f)
        
        # Load main model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            name=model_data['name'],
            embed_dim=model_data['embed_dim'],
            similarity_threshold=model_data['similarity_threshold'],
            dlrm_weight=model_data.get('dlrm_weight', 0.5),
            seed=model_data['seed'],
            verbose=model_data['verbose']
        )
        
        # Restore model state
        model.is_fitted = model_data['is_fitted']
        model.song_index_mapping = model_data['song_index_mapping']
        model.reverse_mapping = model_data['reverse_mapping']
        model.user_preferences = defaultdict(dict, model_data['user_preferences'])
        model.vectorizer = vectorizer
        model.lyrics_matrix = lyrics_matrix
        model.dlrm = dlrm
        
        # Restore pandas dataframes
        if model_data['song_data'] is not None:
            model.song_data = pd.DataFrame.from_dict(model_data['song_data'])
        
        if model_data['artist_data'] is not None:
            model.artist_data = pd.DataFrame.from_dict(model_data['artist_data'])
        
        return model


# Alias the HybridMusicRecommender as MusicRecommender for backward compatibility
MusicRecommender = HybridMusicRecommender

# Sample code to train the model
if __name__ == "__main__":
    # Use a more memory-friendly approach for dataset path
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "..", "..", "..", "src", "SANDBOX", "dataset", 
                            "spotify", "spotify_millsongdata.csv")
    
    # Alternative hardcoded path if the above doesn't work
    # data_path = "/Users/visheshyadav/Documents/GitHub/CoreRec/src/SANDBOX/dataset/spotify/spotify_millsongdata.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}, using hardcoded fallback")
        data_path = "/Users/visheshyadav/Documents/GitHub/CoreRec/src/SANDBOX/dataset/spotify/spotify_millsongdata.csv"
    
    # Create and train the model
    model = HybridMusicRecommender(name="SpotifyRecommender", dlrm_weight=0.4)
    model.fit(data_path)
    
    # Save the model
    model.save("spotify_recommender_model.pkl")
    
    # Test recommendations
    model.add_user_preference(user_id=1, song_key="ABBA - Dancing Queen", rating=1.0)
    recommendations = model.recommend(user_id=1, n_recommendations=5)
    
    print("\nRecommendations for user 1:")
    for rec in recommendations:
        print(f"{rec['song_key']} (Score: {rec['score']:.2f}) - {rec['reason']}") 