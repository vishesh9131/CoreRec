"""
Netflix-like Streaming Platform Frontend

This module provides a Netflix-inspired interface for demonstrating
movie and TV show recommendation models built with CoreRec.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta
from ..base_frontend import BaseFrontend


class NetflixFrontend(BaseFrontend):
    """
    Netflix-inspired frontend for movie/TV recommendation demonstrations.
    """
    
    def __init__(self, recommendation_engine=None, data_path: str = None):
        # Netflix color theme
        theme_config = {
            'primary_color': '#E50914',
            'background_color': '#141414',
            'text_color': '#FFFFFF',
            'secondary_color': '#F40612',
            'accent_color': '#00D4FF',
            'card_color': '#1E1E1E'
        }
        
        super().__init__(
            platform_name="Netflix",
            recommendation_engine=recommendation_engine,
            data_path=data_path,
            theme_config=theme_config
        )
    
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load movie/TV show data and user interactions."""
        if not self.data_path or not os.path.exists(self.data_path):
            return self._generate_sample_content_data()
        
        try:
            content_df = pd.read_csv(self.data_path)
            return content_df, None
        except Exception as e:
            self.logger.warning(f"Could not load data from {self.data_path}: {e}")
            return self._generate_sample_content_data()
    
    def _generate_sample_content_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Generate sample movie and TV show data."""
        # Sample data
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary', 'Animation', 'Crime']
        ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-Y', 'TV-PG', 'TV-14', 'TV-MA']
        
        # Movie titles
        movie_templates = [
            "The {adjective} {noun}",
            "{noun} of {place}",
            "Return to {place}",
            "The Last {noun}",
            "{adjective} {noun} 2",
            "Beyond the {noun}"
        ]
        
        # TV show titles  
        tv_templates = [
            "{adjective} {noun}",
            "The {noun} Chronicles",
            "{place} {noun}",
            "Masters of {noun}"
        ]
        
        adjectives = ['Dark', 'Secret', 'Hidden', 'Lost', 'Ancient', 'Modern', 'Wild', 'Brave', 'Silent', 'Golden']
        nouns = ['Kingdom', 'Mystery', 'Adventure', 'Journey', 'War', 'Love', 'Dreams', 'Shadows', 'Light', 'Truth']
        places = ['New York', 'Tokyo', 'London', 'Mars', 'Atlantis', 'Paradise', 'Hell', 'Future', 'Past', 'Space']
        
        content_data = []
        for i in range(800):
            content_id = f"content_{i:04d}"
            content_type = random.choice(['Movie', 'TV Show'])
            
            # Generate title
            if content_type == 'Movie':
                template = random.choice(movie_templates)
            else:
                template = random.choice(tv_templates)
                
            title = template.format(
                adjective=random.choice(adjectives),
                noun=random.choice(nouns),
                place=random.choice(places)
            )
            
            genre = random.choice(genres)
            rating = random.choice(ratings)
            
            # Duration/seasons
            if content_type == 'Movie':
                duration_minutes = random.randint(80, 180)
                seasons = None
                episodes = None
            else:
                duration_minutes = random.randint(20, 60)  # Per episode
                seasons = random.randint(1, 8)
                episodes = random.randint(6, 24) * seasons
            
            release_year = random.randint(1980, 2024)
            imdb_rating = round(random.uniform(5.0, 9.5), 1)
            netflix_score = random.randint(70, 98)
            
            content_data.append({
                'content_id': content_id,
                'title': title,
                'type': content_type,
                'genre': genre,
                'rating': rating,
                'release_year': release_year,
                'duration_minutes': duration_minutes,
                'seasons': seasons,
                'episodes': episodes,
                'imdb_rating': imdb_rating,
                'netflix_score': netflix_score,
                'director': f"Director {random.randint(1, 100)}",
                'cast': f"Actor {random.randint(1, 50)}, Actor {random.randint(51, 100)}",
                'description': f"An amazing {genre.lower()} {content_type.lower()} that will keep you entertained."
            })
        
        content_df = pd.DataFrame(content_data)
        return content_df, None
    
    def apply_custom_css(self):
        """Apply Netflix-specific CSS styling."""
        st.markdown(f"""
        <style>
        .main {{
            background-color: {self.theme_config['background_color']};
            color: {self.theme_config['text_color']};
            font-family: 'Netflix Sans', Arial, sans-serif;
        }}
        
        .netflix-header {{
            background: linear-gradient(135deg, {self.theme_config['primary_color']}, {self.theme_config['secondary_color']});
            color: {self.theme_config['text_color']};
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .content-card {{
            background-color: {self.theme_config['card_color']};
            color: {self.theme_config['text_color']};
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.6);
            border: 1px solid #333;
            transition: all 0.3s ease;
        }}
        
        .content-card:hover {{
            background-color: #2A2A2A;
            transform: scale(1.03);
            box-shadow: 0 6px 24px rgba(0,0,0,0.8);
        }}
        
        .content-title {{
            font-size: 1.3rem;
            font-weight: bold;
            color: {self.theme_config['text_color']};
            margin-bottom: 8px;
        }}
        
        .content-meta {{
            font-size: 0.9rem;
            color: #CCC;
            margin-bottom: 12px;
        }}
        
        .content-description {{
            font-size: 0.9rem;
            color: #AAA;
            margin-bottom: 16px;
            line-height: 1.4;
        }}
        
        .genre-badge {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 8px;
        }}
        
        .rating-badge {{
            background-color: #FFD700;
            color: #000;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .netflix-score {{
            background-color: #00C851;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .watch-button {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 12px 24px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .watch-button:hover {{
            background-color: {self.theme_config['secondary_color']};
            transform: scale(1.05);
        }}
        
        .my-list-button {{
            background: none;
            border: 2px solid #FFF;
            color: white;
            border-radius: 4px;
            padding: 10px 20px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .my-list-button:hover {{
            background-color: rgba(255,255,255,0.1);
        }}
        
        .thumbs-up, .thumbs-down {{
            background: none;
            border: 2px solid #666;
            color: #666;
            border-radius: 50%;
            padding: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            margin: 0 4px;
        }}
        
        .thumbs-up:hover {{
            border-color: #00C851;
            color: #00C851;
        }}
        
        .thumbs-down:hover {{
            border-color: #FF4444;
            color: #FF4444;
        }}
        
        .liked {{
            border-color: #00C851;
            color: #00C851;
            background-color: rgba(0, 200, 81, 0.1);
        }}
        
        .disliked {{
            border-color: #FF4444;
            color: #FF4444;
            background-color: rgba(255, 68, 68, 0.1);
        }}
        
        .featured {{
            background: linear-gradient(90deg, rgba(229,9,20,0.8), rgba(0,0,0,0.8)), url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><text y="15" font-size="10" fill="white">FEATURED</text></svg>');
            background-size: cover;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: white;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render Netflix-like header."""
        st.markdown(f"""
        <div class="netflix-header">
            <h1>🎬 Netflix Demo</h1>
            <h3>Powered by CoreRec</h3>
            <p>Unlimited movies, TV shows and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render Netflix-like sidebar controls."""
        st.markdown("### 🎬 Browse")
        
        # Content type filter
        content_types = st.multiselect("Content Type", ['Movie', 'TV Show'], default=['Movie', 'TV Show'])
        self.set_session_value('content_types', content_types)
        
        # Genre filter
        if hasattr(self, 'items_df'):
            genres = ['All'] + sorted(self.items_df['genre'].unique().tolist())
            selected_genre = st.selectbox("Genre", genres)
            self.set_session_value('selected_genre', selected_genre)
        
        # Release year filter
        year_range = st.slider("Release Year", 1980, 2024, (2000, 2024))
        self.set_session_value('year_range', year_range)
        
        # Rating filter
        min_rating = st.slider("Minimum IMDB Rating", 5.0, 9.5, 6.0, 0.1)
        self.set_session_value('min_rating', min_rating)
        
        # Sort by
        sort_by = st.selectbox("Sort by", [
            "Recommended", "Netflix Score", "IMDB Rating", "Release Year", "Title"
        ])
        self.set_session_value('sort_by', sort_by)
        
        # Number of recommendations
        num_recs = st.slider("Number of Items", 1, 20, 15)
        self.set_session_value('num_recommendations', num_recs)
    
    def render_main_content(self):
        """Render main content area."""
        user_id = self.get_session_value('user_id', 1)
        
        # Featured content
        self._render_featured_content()
        
        # Get and display recommendations
        recommendations = self.get_recommendations(user_id)
        
        st.markdown("### 🎯 Recommended for You")
        
        if not recommendations:
            st.info("🎬 Start rating content to get personalized recommendations!")
            recommendations = self._get_popular_content(15)
        
        # Display content
        for i, content in enumerate(recommendations):
            self.render_item_card(content, i)
    
    def _render_featured_content(self):
        """Render featured content section."""
        if hasattr(self, 'items_df') and len(self.items_df) > 0:
            featured = self.items_df.nlargest(1, 'netflix_score').iloc[0]
            
            st.markdown(f"""
            <div class="featured">
                <h2>🌟 Featured Today</h2>
                <h3>{featured['title']}</h3>
                <p>{featured['type']} • {featured['release_year']} • {featured['genre']}</p>
                <p>{featured['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_item_card(self, item: Dict, index: int = 0) -> bool:
        """Render a content card with Netflix-like styling."""
        content_id = item['content_id']
        
        # Check if content is liked/disliked
        liked_content = self.get_session_value('liked_items', set())
        disliked_content = self.get_session_value('disliked_items', set())
        
        is_liked = content_id in liked_content
        is_disliked = content_id in disliked_content
        
        # Format duration/episodes
        if item['type'] == 'Movie':
            duration_info = f"{item['duration_minutes']} min"
        else:
            duration_info = f"{item['seasons']} Season{'s' if item['seasons'] > 1 else ''} • {item['episodes']} Episodes"
        
        # Content card
        st.markdown(f"""
        <div class="content-card">
            <div class="content-title">{item['title']}</div>
            <div class="content-meta">
                <span class="genre-badge">{item['genre']}</span>
                <span class="rating-badge">⭐ {item['imdb_rating']}</span>
                <span class="netflix-score">{item['netflix_score']}% Match</span>
            </div>
            <div class="content-meta">
                {item['type']} • {item['release_year']} • {duration_info} • {item['rating']}
            </div>
            <div class="content-description">
                {item['description']}
            </div>
            <div style="margin-top: 8px;">
                <strong>Cast:</strong> {item['cast']}<br>
                <strong>Director:</strong> {item['director']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 0.8, 0.8, 1])
        
        with col1:
            if st.button("▶️ Play", key=f"play_{index}", help="Play content"):
                self._play_content(item)
        
        with col2:
            if st.button("➕ My List", key=f"list_{index}", help="Add to My List"):
                st.success("Added to My List!")
        
        with col3:
            like_class = "liked" if is_liked else ""
            if st.button("👍", key=f"like_{index}", help="Like"):
                self.record_interaction(content_id, 'like')
                st.rerun()
        
        with col4:
            dislike_class = "disliked" if is_disliked else ""
            if st.button("👎", key=f"dislike_{index}", help="Dislike"):
                self.record_interaction(content_id, 'dislike')
                st.rerun()
        
        with col5:
            if st.button("ℹ️ Info", key=f"info_{index}", help="More Info"):
                self._show_content_info(item)
        
        return is_liked or is_disliked
    
    def _play_content(self, content: Dict):
        """Simulate playing content."""
        self.record_interaction(content['content_id'], 'play')
        st.success(f"🎬 Now playing: {content['title']}")
        
        # Show player interface
        st.markdown(f"""
        <div style="background: #000; color: white; padding: 3rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
            <h2>🎬 {content['title']}</h2>
            <p>{content['type']} • {content['genre']} • {content['release_year']}</p>
            <div style="background: #333; padding: 5rem; margin: 2rem 0; border-radius: 4px;">
                📺 Playing...
            </div>
            <p>⭐ IMDB: {content['imdb_rating']} | Netflix Score: {content['netflix_score']}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_content_info(self, content: Dict):
        """Show detailed content information."""
        st.info(f"""
        **{content['title']}** ({content['release_year']})
        
        **Type:** {content['type']}
        **Genre:** {content['genre']}
        **Rating:** {content['rating']}
        **IMDB Rating:** ⭐ {content['imdb_rating']}
        **Netflix Score:** {content['netflix_score']}% Match
        
        **Cast:** {content['cast']}
        **Director:** {content['director']}
        
        **Description:** {content['description']}
        """)
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 15) -> List[Dict]:
        """Get content recommendations for a user."""
        if not hasattr(self, 'items_df'):
            return []
        
        # Get user preferences
        liked_content = self.get_session_value('liked_items', set())
        disliked_content = self.get_session_value('disliked_items', set())
        content_types = self.get_session_value('content_types', ['Movie', 'TV Show'])
        selected_genre = self.get_session_value('selected_genre', 'All')
        year_range = self.get_session_value('year_range', (2000, 2024))
        min_rating = self.get_session_value('min_rating', 6.0)
        sort_by = self.get_session_value('sort_by', 'Recommended')
        num_recs = self.get_session_value('num_recommendations', num_recommendations)
        
        # Filter content
        df = self.items_df.copy()
        
        # Remove already interacted content
        all_interacted = liked_content.union(disliked_content)
        if all_interacted:
            df = df[~df['content_id'].isin(all_interacted)]
        
        # Apply filters
        if content_types:
            df = df[df['type'].isin(content_types)]
        
        if selected_genre != 'All':
            df = df[df['genre'] == selected_genre]
        
        df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]
        df = df[df['imdb_rating'] >= min_rating]
        
        # Apply sorting
        if sort_by == "Netflix Score":
            df = df.sort_values('netflix_score', ascending=False)
        elif sort_by == "IMDB Rating":
            df = df.sort_values('imdb_rating', ascending=False)
        elif sort_by == "Release Year":
            df = df.sort_values('release_year', ascending=False)
        elif sort_by == "Title":
            df = df.sort_values('title')
        else:  # Recommended
            if liked_content:
                df = self._get_content_based_recommendations(df, liked_content)
            else:
                # No preferences, show by Netflix score
                df = df.sort_values('netflix_score', ascending=False)
        
        # Return top recommendations
        recommendations = df.head(num_recs).to_dict('records')
        return recommendations
    
    def _get_content_based_recommendations(self, df: pd.DataFrame, liked_content: set) -> pd.DataFrame:
        """Get content-based recommendations."""
        if df.empty:
            return df
        
        # Get liked content features
        liked_df = self.items_df[self.items_df['content_id'].isin(liked_content)]
        
        if not liked_df.empty:
            # Get preferred genres and types
            preferred_genres = liked_df['genre'].value_counts()
            preferred_types = liked_df['type'].value_counts()
            
            # Calculate average ratings
            avg_imdb = liked_df['imdb_rating'].mean()
            avg_year = liked_df['release_year'].mean()
            
            # Calculate similarity scores
            df['genre_boost'] = df['genre'].map(
                lambda x: preferred_genres.get(x, 0) / len(liked_content)
            ).fillna(0)
            
            df['type_boost'] = df['type'].map(
                lambda x: preferred_types.get(x, 0) / len(liked_content)
            ).fillna(0)
            
            df['rating_similarity'] = 1 - abs(df['imdb_rating'] - avg_imdb) / 5.0
            df['year_similarity'] = 1 - abs(df['release_year'] - avg_year) / 50.0
            
            # Overall similarity score
            df['similarity_score'] = (
                df['genre_boost'] * 0.4 +
                df['type_boost'] * 0.2 +
                df['rating_similarity'] * 0.2 +
                df['year_similarity'] * 0.1 +
                (df['netflix_score'] / 100) * 0.1
            )
            
            # Sort by similarity
            df = df.sort_values('similarity_score', ascending=False)
        
        return df
    
    def _get_popular_content(self, num_content: int = 15) -> List[Dict]:
        """Get popular content for new users."""
        if not hasattr(self, 'items_df'):
            return []
        
        return self.items_df.nlargest(num_content, 'netflix_score').to_dict('records')


def generate_sample_data(file_path: str, num_content: int = 800):
    """Generate sample content data file."""
    frontend = NetflixFrontend()
    
    # Generate custom amount of content
    import random
    import pandas as pd
    
    # Sample data
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Documentary', 'Animation', 'Crime']
    ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'TV-Y', 'TV-PG', 'TV-14', 'TV-MA']
    
    # Movie titles
    movie_templates = [
        "The {adjective} {noun}",
        "{noun} of {place}",
        "Return to {place}",
        "The Last {noun}",
        "{adjective} {noun} 2",
        "Beyond the {noun}"
    ]
    
    # TV show titles  
    tv_templates = [
        "{adjective} {noun}",
        "The {noun} Chronicles",
        "{place} {noun}",
        "Masters of {noun}"
    ]
    
    adjectives = ['Dark', 'Secret', 'Hidden', 'Lost', 'Ancient', 'Modern', 'Wild', 'Brave', 'Silent', 'Golden']
    nouns = ['Kingdom', 'Mystery', 'Adventure', 'Journey', 'War', 'Love', 'Dreams', 'Shadows', 'Light', 'Truth']
    places = ['New York', 'Tokyo', 'London', 'Mars', 'Atlantis', 'Paradise', 'Hell', 'Future', 'Past', 'Space']
    
    content_data = []
    for i in range(num_content):
        content_id = f"content_{i:04d}"
        content_type = random.choice(['Movie', 'TV Show'])
        
        # Generate title
        if content_type == 'Movie':
            template = random.choice(movie_templates)
        else:
            template = random.choice(tv_templates)
            
        title = template.format(
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            place=random.choice(places)
        )
        
        genre = random.choice(genres)
        rating = random.choice(ratings)
        
        # Duration/seasons
        if content_type == 'Movie':
            duration_minutes = random.randint(80, 180)
            seasons = None
            episodes = None
        else:
            duration_minutes = random.randint(20, 60)  # Per episode
            seasons = random.randint(1, 8)
            episodes = random.randint(6, 24) * seasons
        
        release_year = random.randint(1980, 2024)
        imdb_rating = round(random.uniform(5.0, 9.5), 1)
        netflix_score = random.randint(70, 98)
        
        content_data.append({
            'content_id': content_id,
            'title': title,
            'type': content_type,
            'genre': genre,
            'rating': rating,
            'release_year': release_year,
            'duration_minutes': duration_minutes,
            'seasons': seasons,
            'episodes': episodes,
            'imdb_rating': imdb_rating,
            'netflix_score': netflix_score,
            'director': f"Director {random.randint(1, 100)}",
            'cast': f"Actor {random.randint(1, 50)}, Actor {random.randint(51, 100)}",
            'description': f"An amazing {genre.lower()} {content_type.lower()} that will keep you entertained."
        })
    
    content_df = pd.DataFrame(content_data)
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    content_df.to_csv(file_path, index=False)
    print(f"Generated {num_content} sample content items at {file_path}") 