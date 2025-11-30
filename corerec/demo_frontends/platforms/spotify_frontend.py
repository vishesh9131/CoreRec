"""
Spotify-like Music Streaming Platform Frontend

This module provides a Spotify-inspired interface for demonstrating
music recommendation models built with CoreRec.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta
from ..base_frontend import BaseFrontend


class SpotifyFrontend(BaseFrontend):
    """
    Spotify-inspired frontend for music recommendation demonstrations.
    """

    def __init__(self, recommendation_engine=None, data_path: str = None):
        # Spotify color theme
        theme_config = {
            "primary_color": "#1DB954",
            "background_color": "#191414",
            "text_color": "#FFFFFF",
            "secondary_color": "#1ED760",
            "accent_color": "#FF6B6B",
            "card_color": "#282828",
        }

        super().__init__(
            platform_name="Spotify",
            recommendation_engine=recommendation_engine,
            data_path=data_path,
            theme_config=theme_config,
        )

    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load music tracks data and user interactions."""
        # If data file doesn't exist, generate sample data
        if not self.data_path or not os.path.exists(self.data_path):
            return self._generate_sample_music_data()

        try:
            tracks_df = pd.read_csv(self.data_path)
            return tracks_df, None
        except Exception as e:
            self.logger.warning(f"Could not load data from {self.data_path}: {e}")
            return self._generate_sample_music_data()

    def _generate_sample_music_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Generate sample music data for demonstration."""
        # Sample artists and genres
        artists = [
            "The Beatles",
            "Taylor Swift",
            "Drake",
            "Ed Sheeran",
            "Billie Eilish",
            "Ariana Grande",
            "Post Malone",
            "The Weeknd",
            "Dua Lipa",
            "Harry Styles",
            "Olivia Rodrigo",
            "Justin Bieber",
            "BTS",
            "Bad Bunny",
            "Adele",
            "Eminem",
            "Kendrick Lamar",
            "Rihanna",
            "Bruno Mars",
            "Lady Gaga",
        ]

        genres = [
            "Pop",
            "Rock",
            "Hip Hop",
            "R&B",
            "Electronic",
            "Country",
            "Jazz",
            "Classical",
            "Reggae",
            "Folk",
            "Alternative",
            "Indie",
            "Blues",
            "Punk",
            "Metal",
            "Funk",
            "Soul",
            "Disco",
            "House",
            "Techno",
        ]

        # Sample track names
        track_adjectives = [
            "Beautiful",
            "Dark",
            "Electric",
            "Golden",
            "Wild",
            "Sweet",
            "Blue",
            "Red",
            "New",
            "Old",
        ]
        track_nouns = [
            "Love",
            "Dreams",
            "Night",
            "Heart",
            "Soul",
            "Fire",
            "Rain",
            "Sun",
            "Moon",
            "Star",
        ]

        # Generate tracks
        tracks_data = []
        for i in range(500):
            track_id = f"track_{i:04d}"
            artist = random.choice(artists)
            genre = random.choice(genres)

            # Generate track name
            if random.random() < 0.7:
                track_name = f"{random.choice(track_adjectives)} {random.choice(track_nouns)}"
            else:
                track_name = f"{random.choice(track_nouns)} {random.choice(track_adjectives)}"

            # Add some variety to names
            if random.random() < 0.3:
                track_name += f" (feat. {random.choice(artists)})"

            duration_ms = random.randint(120000, 360000)  # 2-6 minutes
            popularity = random.randint(0, 100)
            energy = random.random()
            valence = random.random()
            danceability = random.random()

            # Release date
            release_date = datetime.now() - timedelta(days=random.randint(0, 3650))

            tracks_data.append(
                {
                    "track_id": track_id,
                    "track_name": track_name,
                    "artist_name": artist,
                    "genre": genre,
                    "duration_ms": duration_ms,
                    "popularity": popularity,
                    "energy": energy,
                    "valence": valence,
                    "danceability": danceability,
                    "release_date": release_date.strftime("%Y-%m-%d"),
                    "album_name": f"{track_name} - Single"
                    if random.random() < 0.4
                    else f"{artist}'s Album",
                    "explicit": random.random() < 0.2,
                }
            )

        tracks_df = pd.DataFrame(tracks_data)
        return tracks_df, None

    def apply_custom_css(self):
        """Apply Spotify-specific CSS styling."""
        st.markdown(
            f"""
        <style>
        /* Import Spotify-like font */
        @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@300;400;500;700&display=swap');
        
        .main {{
            background-color: {self.theme_config['background_color']};
            color: {self.theme_config['text_color']};
            font-family: 'Circular Std', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        .spotify-header {{
            background: linear-gradient(135deg, {self.theme_config['primary_color']}, {self.theme_config['secondary_color']});
            color: {self.theme_config['text_color']};
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .track-card {{
            background-color: {self.theme_config['card_color']};
            color: {self.theme_config['text_color']};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 1px solid #404040;
            transition: all 0.2s ease;
        }}
        
        .track-card:hover {{
            background-color: #3E3E3E;
            transform: scale(1.02);
        }}
        
        .track-title {{
            font-size: 1.1rem;
            font-weight: 500;
            color: {self.theme_config['text_color']};
            margin-bottom: 4px;
        }}
        
        .track-artist {{
            font-size: 0.9rem;
            color: #B3B3B3;
            margin-bottom: 8px;
        }}
        
        .track-details {{
            font-size: 0.8rem;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .genre-tag {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 500;
        }}
        
        .play-button {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .play-button:hover {{
            background-color: {self.theme_config['secondary_color']};
            transform: scale(1.1);
        }}
        
        .like-button, .dislike-button {{
            border: none;
            background: none;
            font-size: 1.2rem;
            cursor: pointer;
            padding: 8px;
            border-radius: 50%;
            transition: all 0.2s ease;
        }}
        
        .like-button:hover {{
            background-color: rgba(29, 185, 84, 0.1);
        }}
        
        .dislike-button:hover {{
            background-color: rgba(255, 107, 107, 0.1);
        }}
        
        .now-playing {{
            background: linear-gradient(90deg, {self.theme_config['primary_color']}, {self.theme_config['secondary_color']});
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        .sidebar .stSelectbox > div > div {{
            background-color: {self.theme_config['card_color']};
            color: {self.theme_config['text_color']};
        }}
        
        .stButton > button {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            border: none;
            border-radius: 24px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {self.theme_config['secondary_color']};
            transform: scale(1.05);
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

    def render_header(self):
        """Render Spotify-like header."""
        st.markdown(
            f"""
        <div class="spotify-header">
            <h1>🎵 Spotify Demo</h1>
            <h3>Powered by CoreRec</h3>
            <p>Discover your next favorite song with AI-powered recommendations</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        """Render Spotify-like sidebar controls."""
        st.markdown("### 🎵 Music Controls")

        # Genre filter
        if hasattr(self, "items_df"):
            genres = ["All"] + sorted(self.items_df["genre"].unique().tolist())
            selected_genre = st.selectbox("Filter by Genre", genres)
            self.set_session_value("selected_genre", selected_genre)

        # Mood filter
        mood = st.selectbox(
            "Mood",
            [
                "Any",
                "Happy (High Valence)",
                "Sad (Low Valence)",
                "Energetic (High Energy)",
                "Chill (Low Energy)",
                "Danceable",
                "Acoustic",
            ],
        )
        self.set_session_value("selected_mood", mood)

        # Number of recommendations
        num_recs = st.slider("Number of Recommendations", 1, 20, 10)
        self.set_session_value("num_recommendations", num_recs)

        # Shuffle button
        if st.button("🔀 Shuffle Recommendations"):
            self.set_session_value("shuffle_seed", random.randint(1, 1000))
            st.rerun()

    def render_main_content(self):
        """Render main content area with recommendations."""
        # Get current user and preferences
        user_id = self.get_session_value("user_id", 1)

        # Show currently "playing" track
        self._render_now_playing()

        # Get and display recommendations
        recommendations = self.get_recommendations(user_id)

        st.markdown("### 🎯 Recommended for You")

        if not recommendations:
            st.info("🎵 Start liking some tracks to get personalized recommendations!")
            # Show some popular tracks to get started
            recommendations = self._get_popular_tracks(10)

        # Display tracks
        for i, track in enumerate(recommendations):
            self.render_item_card(track, i)

    def _render_now_playing(self):
        """Render the currently playing track section."""
        # Simulate a currently playing track
        if hasattr(self, "items_df") and len(self.items_df) > 0:
            if not hasattr(st.session_state, "now_playing_track"):
                st.session_state.now_playing_track = self.items_df.sample(1).iloc[0]

            track = st.session_state.now_playing_track

            st.markdown(
                f"""
            <div class="now-playing">
                <h4>🎵 Now Playing</h4>
                <h3>{track['track_name']}</h3>
                <p>by {track['artist_name']} • {track['genre']}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_item_card(self, item: Dict, index: int = 0) -> bool:
        """Render a track card with Spotify-like styling."""
        track_id = item["track_id"]

        # Check if track is liked/disliked
        liked_tracks = self.get_session_value("liked_items", set())
        disliked_tracks = self.get_session_value("disliked_items", set())

        is_liked = track_id in liked_tracks
        is_disliked = track_id in disliked_tracks

        # Format duration
        duration_str = self._format_duration(item["duration_ms"])

        # Create columns for track layout
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])

        with col1:
            # Play button
            if st.button("▶️", key=f"play_{index}", help="Play track"):
                self._play_track(item)

        with col2:
            # Track info
            st.markdown(
                f"""
            <div class="track-card">
                <div class="track-title">{item['track_name']}</div>
                <div class="track-artist">{item['artist_name']}</div>
                <div class="track-details">
                    <span class="genre-tag">{item['genre']}</span>
                    <span>Popularity: {item['popularity']}/100</span>
                    <span>{duration_str}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            # Like/dislike buttons
            col3a, col3b = st.columns(2)

            with col3a:
                like_emoji = "❤️" if is_liked else "🤍"
                if st.button(like_emoji, key=f"like_{index}", help="Like this track"):
                    self.record_interaction(track_id, "like")
                    st.rerun()

            with col3b:
                dislike_emoji = "👎" if is_disliked else "👍"
                if st.button(dislike_emoji, key=f"dislike_{index}", help="Dislike this track"):
                    self.record_interaction(track_id, "dislike")
                    st.rerun()

        return is_liked or is_disliked

    def _play_track(self, track: Dict):
        """Simulate playing a track."""
        st.session_state.now_playing_track = track
        self.record_interaction(track["track_id"], "play")
        st.success(f"🎵 Playing: {track['track_name']} by {track['artist_name']}")

    def _format_duration(self, duration_ms: int) -> str:
        """Format duration from milliseconds to MM:SS format."""
        duration_s = duration_ms // 1000
        minutes = duration_s // 60
        seconds = duration_s % 60
        return f"{minutes}:{seconds:02d}"

    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """Get track recommendations for a user."""
        if not hasattr(self, "items_df"):
            return []

        # Get user preferences
        liked_tracks = self.get_session_value("liked_items", set())
        disliked_tracks = self.get_session_value("disliked_items", set())
        selected_genre = self.get_session_value("selected_genre", "All")
        selected_mood = self.get_session_value("selected_mood", "Any")
        num_recs = self.get_session_value("num_recommendations", num_recommendations)

        # Filter tracks
        df = self.items_df.copy()

        # Remove already interacted tracks
        all_interacted = liked_tracks.union(disliked_tracks)
        if all_interacted:
            df = df[~df["track_id"].isin(all_interacted)]

        # Apply genre filter
        if selected_genre != "All":
            df = df[df["genre"] == selected_genre]

        # Apply mood filter
        if selected_mood != "Any":
            if selected_mood == "Happy (High Valence)":
                df = df[df["valence"] > 0.7]
            elif selected_mood == "Sad (Low Valence)":
                df = df[df["valence"] < 0.3]
            elif selected_mood == "Energetic (High Energy)":
                df = df[df["energy"] > 0.7]
            elif selected_mood == "Chill (Low Energy)":
                df = df[df["energy"] < 0.3]
            elif selected_mood == "Danceable":
                df = df[df["danceability"] > 0.7]

        # If we have liked tracks, try to recommend similar ones
        if liked_tracks and self.recommendation_engine:
            # Use the recommendation engine if available
            try:
                recommendations = self._get_engine_recommendations(user_id, num_recs)
                if recommendations:
                    return recommendations
            except Exception as e:
                self.logger.warning(f"Engine recommendations failed: {e}")

        # Fallback to content-based recommendations
        return self._get_content_based_recommendations(df, liked_tracks, num_recs)

    def _get_engine_recommendations(self, user_id: int, num_recs: int) -> List[Dict]:
        """Get recommendations using the CoreRec engine."""
        if not self.recommendation_engine:
            return []

        try:
            # This would depend on the specific CoreRec engine interface
            # For now, return empty list as we'd need the actual engine
            return []
        except Exception as e:
            self.logger.error(f"Engine recommendation error: {e}")
            return []

    def _get_content_based_recommendations(
        self, df: pd.DataFrame, liked_tracks: set, num_recs: int
    ) -> List[Dict]:
        """Get content-based recommendations."""
        if df.empty:
            return []

        if liked_tracks:
            # Get liked tracks features
            liked_df = self.items_df[self.items_df["track_id"].isin(liked_tracks)]

            if not liked_df.empty:
                # Calculate average features of liked tracks
                feature_cols = ["energy", "valence", "danceability", "popularity"]
                avg_features = liked_df[feature_cols].mean()

                # Calculate similarity to average preferences
                for col in feature_cols:
                    df[f"{col}_diff"] = abs(df[col] - avg_features[col])

                # Score based on similarity (lower difference = higher score)
                df["similarity_score"] = (
                    (1 - df["energy_diff"]) * 0.3
                    + (1 - df["valence_diff"]) * 0.3
                    + (1 - df["danceability_diff"]) * 0.2
                    + (df["popularity"] / 100) * 0.2  # Boost popular tracks
                )

                # Sort by similarity score
                df = df.sort_values("similarity_score", ascending=False)
        else:
            # No preferences yet, show popular tracks
            df = df.sort_values("popularity", ascending=False)

        # Add shuffle if enabled
        shuffle_seed = self.get_session_value("shuffle_seed", None)
        if shuffle_seed:
            df = df.sample(frac=1, random_state=shuffle_seed)

        # Return top recommendations
        recommendations = df.head(num_recs).to_dict("records")
        return recommendations

    def _get_popular_tracks(self, num_tracks: int = 10) -> List[Dict]:
        """Get popular tracks for new users."""
        if not hasattr(self, "items_df"):
            return []

        return self.items_df.nlargest(num_tracks, "popularity").to_dict("records")


def generate_sample_data(file_path: str, num_tracks: int = 500):
    """Generate sample music data file."""
    frontend = SpotifyFrontend()
    # Update the data generation to use the provided number
    original_method = frontend._generate_sample_music_data

    def custom_generate():
        # Temporarily modify the range in the generation
        import random

        # Sample artists and genres (same as before)
        artists = [
            "The Beatles",
            "Taylor Swift",
            "Drake",
            "Ed Sheeran",
            "Billie Eilish",
            "Ariana Grande",
            "Post Malone",
            "The Weeknd",
            "Dua Lipa",
            "Harry Styles",
            "Olivia Rodrigo",
            "Justin Bieber",
            "BTS",
            "Bad Bunny",
            "Adele",
            "Eminem",
            "Kendrick Lamar",
            "Rihanna",
            "Bruno Mars",
            "Lady Gaga",
        ]

        genres = [
            "Pop",
            "Rock",
            "Hip Hop",
            "R&B",
            "Electronic",
            "Country",
            "Jazz",
            "Classical",
            "Reggae",
            "Folk",
            "Alternative",
            "Indie",
            "Blues",
            "Punk",
            "Metal",
            "Funk",
            "Soul",
            "Disco",
            "House",
            "Techno",
        ]

        # Sample track names
        track_adjectives = [
            "Beautiful",
            "Dark",
            "Electric",
            "Golden",
            "Wild",
            "Sweet",
            "Blue",
            "Red",
            "New",
            "Old",
        ]
        track_nouns = [
            "Love",
            "Dreams",
            "Night",
            "Heart",
            "Soul",
            "Fire",
            "Rain",
            "Sun",
            "Moon",
            "Star",
        ]

        # Generate tracks with custom count
        tracks_data = []
        for i in range(num_tracks):
            track_id = f"track_{i:04d}"
            artist = random.choice(artists)
            genre = random.choice(genres)

            # Generate track name
            if random.random() < 0.7:
                track_name = f"{random.choice(track_adjectives)} {random.choice(track_nouns)}"
            else:
                track_name = f"{random.choice(track_nouns)} {random.choice(track_adjectives)}"

            # Add some variety to names
            if random.random() < 0.3:
                track_name += f" (feat. {random.choice(artists)})"

            duration_ms = random.randint(120000, 360000)  # 2-6 minutes
            popularity = random.randint(0, 100)
            energy = random.random()
            valence = random.random()
            danceability = random.random()

            # Release date
            from datetime import datetime, timedelta

            release_date = datetime.now() - timedelta(days=random.randint(0, 3650))

            tracks_data.append(
                {
                    "track_id": track_id,
                    "track_name": track_name,
                    "artist_name": artist,
                    "genre": genre,
                    "duration_ms": duration_ms,
                    "popularity": popularity,
                    "energy": energy,
                    "valence": valence,
                    "danceability": danceability,
                    "release_date": release_date.strftime("%Y-%m-%d"),
                    "album_name": f"{track_name} - Single"
                    if random.random() < 0.4
                    else f"{artist}'s Album",
                    "explicit": random.random() < 0.2,
                }
            )

        import pandas as pd

        tracks_df = pd.DataFrame(tracks_data)
        return tracks_df, None

    tracks_df, _ = custom_generate()

    # Ensure directory exists
    import os

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to CSV
    tracks_df.to_csv(file_path, index=False)
    print(f"Generated {num_tracks} sample tracks at {file_path}")
