"""
YouTube-like Video Platform Frontend

This module provides a YouTube-inspired interface for demonstrating
video recommendation models built with CoreRec.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime, timedelta
from ..base_frontend import BaseFrontend


class YouTubeFrontend(BaseFrontend):
    """
    YouTube-inspired frontend for video recommendation demonstrations.
    """
    
    def __init__(self, recommendation_engine=None, data_path: str = None):
        # YouTube color theme
        theme_config = {
            'primary_color': '#FF0000',
            'background_color': '#0F0F0F',
            'text_color': '#FFFFFF',
            'secondary_color': '#FF4444',
            'accent_color': '#00D4FF',
            'card_color': '#1A1A1A'
        }
        
        super().__init__(
            platform_name="YouTube",
            recommendation_engine=recommendation_engine,
            data_path=data_path,
            theme_config=theme_config
        )
    
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load video data and user interactions."""
        if not self.data_path or not os.path.exists(self.data_path):
            return self._generate_sample_video_data()
        
        try:
            videos_df = pd.read_csv(self.data_path)
            return videos_df, None
        except Exception as e:
            self.logger.warning(f"Could not load data from {self.data_path}: {e}")
            return self._generate_sample_video_data()
    
    def _generate_sample_video_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Generate sample video data for demonstration."""
        # Sample channels and categories
        channels = [
            'TechReviews', 'CookingMaster', 'FitnessGuru', 'MusicVibes', 'GameZone',
            'TravelWorld', 'ScienceHub', 'ArtStudio', 'NewsDaily', 'ComedyCentral',
            'EducationPlus', 'DIYProjects', 'FashionTrends', 'AutoReviews', 'PetLovers',
            'BusinessInsights', 'HealthTips', 'MovieReviews', 'SportsCentral', 'TechTalks'
        ]
        
        categories = [
            'Technology', 'Cooking', 'Fitness', 'Music', 'Gaming',
            'Travel', 'Science', 'Art', 'News', 'Comedy',
            'Education', 'DIY', 'Fashion', 'Automotive', 'Pets',
            'Business', 'Health', 'Movies', 'Sports', 'Lifestyle'
        ]
        
        # Sample video title templates
        title_templates = [
            "How to {action} {object}",
            "Top 10 {adjective} {category}",
            "Ultimate Guide to {topic}",
            "Amazing {adjective} {object}",
            "Why {topic} is {adjective}",
            "Best {object} for {purpose}",
            "Incredible {adjective} {topic}",
            "{number} {adjective} {category} Tips"
        ]
        
        actions = ["build", "create", "fix", "improve", "master", "learn", "discover"]
        objects = ["gadgets", "recipes", "workouts", "songs", "games", "places", "experiments"]
        adjectives = ["amazing", "incredible", "ultimate", "best", "awesome", "mind-blowing", "surprising"]
        topics = ["technology", "cooking", "fitness", "music", "gaming", "travel", "science"]
        purposes = ["beginners", "professionals", "home", "work", "travel", "kids", "adults"]
        
        # Generate videos
        videos_data = []
        for i in range(1000):
            video_id = f"video_{i:04d}"
            channel = random.choice(channels)
            category = random.choice(categories)
            
            # Generate title
            template = random.choice(title_templates)
            title = template.format(
                action=random.choice(actions),
                object=random.choice(objects),
                adjective=random.choice(adjectives),
                category=category.lower(),
                topic=random.choice(topics),
                purpose=random.choice(purposes),
                number=random.randint(3, 20)
            )
            
            # Video metrics
            views = random.randint(100, 10000000)
            likes = int(views * random.uniform(0.01, 0.15))
            duration_seconds = random.randint(30, 3600)  # 30 sec to 1 hour
            
            # Upload date
            upload_date = datetime.now() - timedelta(days=random.randint(0, 1095))  # Up to 3 years ago
            
            # Video features
            quality_score = random.random()
            engagement_rate = random.random()
            educational_value = random.random()
            entertainment_value = random.random()
            
            videos_data.append({
                'video_id': video_id,
                'title': title,
                'channel_name': channel,
                'category': category,
                'duration_seconds': duration_seconds,
                'views': views,
                'likes': likes,
                'upload_date': upload_date.strftime('%Y-%m-%d'),
                'quality_score': quality_score,
                'engagement_rate': engagement_rate,
                'educational_value': educational_value,
                'entertainment_value': entertainment_value,
                'thumbnail_url': f"https://img.youtube.com/vi/sample_{i}/maxresdefault.jpg",
                'description': f"An amazing {category.lower()} video from {channel}. {title}"
            })
        
        videos_df = pd.DataFrame(videos_data)
        return videos_df, None
    
    def apply_custom_css(self):
        """Apply YouTube-specific CSS styling."""
        st.markdown(f"""
        <style>
        .main {{
            background-color: {self.theme_config['background_color']};
            color: {self.theme_config['text_color']};
            font-family: 'Roboto', Arial, sans-serif;
        }}
        
        .youtube-header {{
            background: linear-gradient(135deg, {self.theme_config['primary_color']}, {self.theme_config['secondary_color']});
            color: {self.theme_config['text_color']};
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .video-card {{
            background-color: {self.theme_config['card_color']};
            color: {self.theme_config['text_color']};
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
            border: 1px solid #333;
            transition: all 0.2s ease;
        }}
        
        .video-card:hover {{
            background-color: #2A2A2A;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.7);
        }}
        
        .video-title {{
            font-size: 1.2rem;
            font-weight: 500;
            color: {self.theme_config['text_color']};
            margin-bottom: 6px;
            line-height: 1.3;
        }}
        
        .video-channel {{
            font-size: 0.9rem;
            color: #AAA;
            margin-bottom: 8px;
        }}
        
        .video-stats {{
            font-size: 0.8rem;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        
        .category-tag {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 500;
        }}
        
        .thumbnail {{
            width: 100%;
            height: 120px;
            background: linear-gradient(45deg, #333, #555);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 12px;
            color: #999;
            font-size: 2rem;
        }}
        
        .watch-button {{
            background-color: {self.theme_config['primary_color']};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .watch-button:hover {{
            background-color: {self.theme_config['secondary_color']};
            transform: scale(1.05);
        }}
        
        .interaction-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }}
        
        .like-btn, .dislike-btn {{
            background: none;
            border: 1px solid #666;
            color: #666;
            border-radius: 20px;
            padding: 6px 12px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .like-btn:hover {{
            border-color: {self.theme_config['accent_color']};
            color: {self.theme_config['accent_color']};
        }}
        
        .dislike-btn:hover {{
            border-color: #FF6B6B;
            color: #FF6B6B;
        }}
        
        .liked {{
            background-color: {self.theme_config['accent_color']};
            color: white;
            border-color: {self.theme_config['accent_color']};
        }}
        
        .disliked {{
            background-color: #FF6B6B;
            color: white;
            border-color: #FF6B6B;
        }}
        
        .trending {{
            background: linear-gradient(90deg, #FF6B6B, #FFB74D);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render YouTube-like header."""
        st.markdown(f"""
        <div class="youtube-header">
            <h1>📺 YouTube Demo</h1>
            <h3>Powered by CoreRec</h3>
            <p>Discover amazing videos tailored just for you</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render YouTube-like sidebar controls."""
        st.markdown("### 📺 Video Controls")
        
        # Category filter
        if hasattr(self, 'items_df'):
            categories = ['All'] + sorted(self.items_df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category", categories)
            self.set_session_value('selected_category', selected_category)
        
        # Duration filter
        duration_filter = st.selectbox("Video Duration", [
            "Any", "Short (< 4 min)", "Medium (4-20 min)", "Long (> 20 min)"
        ])
        self.set_session_value('duration_filter', duration_filter)
        
        # Quality filter
        quality_filter = st.selectbox("Video Quality", [
            "Any", "High Quality (>0.7)", "Medium Quality (>0.4)", "Any Quality"
        ])
        self.set_session_value('quality_filter', quality_filter)
        
        # Sort by
        sort_by = st.selectbox("Sort by", [
            "Recommended", "Most Views", "Most Recent", "Most Liked", "Trending"
        ])
        self.set_session_value('sort_by', sort_by)
        
        # Number of recommendations
        num_recs = st.slider("Number of Videos", 1, 20, 12)
        self.set_session_value('num_recommendations', num_recs)
    
    def render_main_content(self):
        """Render main content area with video recommendations."""
        user_id = self.get_session_value('user_id', 1)
        
        # Show trending section
        self._render_trending_section()
        
        # Get and display recommendations
        recommendations = self.get_recommendations(user_id)
        
        st.markdown("### 🎯 Recommended Videos")
        
        if not recommendations:
            st.info("📺 Start watching and liking videos to get personalized recommendations!")
            recommendations = self._get_trending_videos(12)
        
        # Display videos in a grid
        cols = st.columns(3)
        for i, video in enumerate(recommendations):
            col_idx = i % 3
            with cols[col_idx]:
                self.render_item_card(video, i)
    
    def _render_trending_section(self):
        """Render trending videos section."""
        trending_videos = self._get_trending_videos(3)
        if trending_videos:
            st.markdown("""
            <div class="trending">
                <h4>🔥 Trending Now</h4>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, video in enumerate(trending_videos):
                with cols[i]:
                    self._render_trending_card(video, i)
    
    def _render_trending_card(self, video: Dict, index: int):
        """Render a smaller trending video card."""
        st.markdown(f"""
        <div style="background: #2A2A2A; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <div class="thumbnail">📹</div>
            <div style="font-size: 0.9rem; font-weight: 500; margin-bottom: 4px;">
                {video['title'][:50]}...
            </div>
            <div style="font-size: 0.8rem; color: #AAA;">
                {video['channel_name']} • {self._format_views(video['views'])} views
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_item_card(self, item: Dict, index: int = 0) -> bool:
        """Render a video card with YouTube-like styling."""
        video_id = item['video_id']
        
        # Check if video is liked/disliked
        liked_videos = self.get_session_value('liked_items', set())
        disliked_videos = self.get_session_value('disliked_items', set())
        
        is_liked = video_id in liked_videos
        is_disliked = video_id in disliked_videos
        
        # Format metrics
        views_str = self._format_views(item['views'])
        duration_str = self._format_duration(item['duration_seconds'])
        upload_date = self._format_upload_date(item['upload_date'])
        
        # Video card
        st.markdown(f"""
        <div class="video-card">
            <div class="thumbnail">📹</div>
            <div class="video-title">{item['title']}</div>
            <div class="video-channel">{item['channel_name']}</div>
            <div class="video-stats">
                <span>{views_str} views • {upload_date}</span>
                <span class="category-tag">{item['category']}</span>
            </div>
            <div class="video-stats">
                <span>Duration: {duration_str}</span>
                <span>👍 {self._format_number(item['likes'])}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            if st.button("▶️ Watch", key=f"watch_{index}", help="Watch video"):
                self._watch_video(item)
        
        with col2:
            like_label = "👍" if not is_liked else "👍✅"
            if st.button(like_label, key=f"like_{index}", help="Like video"):
                self.record_interaction(video_id, 'like')
                st.rerun()
        
        with col3:
            dislike_label = "👎" if not is_disliked else "👎✅"
            if st.button(dislike_label, key=f"dislike_{index}", help="Dislike video"):
                self.record_interaction(video_id, 'dislike')
                st.rerun()
        
        with col4:
            if st.button("💾", key=f"save_{index}", help="Save to Watch Later"):
                st.success("Saved to Watch Later!")
        
        return is_liked or is_disliked
    
    def _watch_video(self, video: Dict):
        """Simulate watching a video."""
        self.record_interaction(video['video_id'], 'watch')
        st.success(f"🎬 Now watching: {video['title']}")
        
        # Show video "player"
        st.markdown(f"""
        <div style="background: #000; color: white; padding: 2rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
            <h3>🎬 {video['title']}</h3>
            <p>Channel: {video['channel_name']}</p>
            <p>Duration: {self._format_duration(video['duration_seconds'])}</p>
            <div style="background: #333; padding: 4rem; margin: 1rem 0; border-radius: 4px;">
                📺 Video Playing...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _format_views(self, views: int) -> str:
        """Format view count for display."""
        if views >= 1000000:
            return f"{views/1000000:.1f}M"
        elif views >= 1000:
            return f"{views/1000:.1f}K"
        else:
            return str(views)
    
    def _format_number(self, number: int) -> str:
        """Format large numbers."""
        if number >= 1000000:
            return f"{number/1000000:.1f}M"
        elif number >= 1000:
            return f"{number/1000:.1f}K"
        else:
            return str(number)
    
    def _format_duration(self, duration_seconds: int) -> str:
        """Format duration from seconds to MM:SS or HH:MM:SS format."""
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def _format_upload_date(self, upload_date: str) -> str:
        """Format upload date for display."""
        try:
            date = datetime.strptime(upload_date, '%Y-%m-%d')
            days_ago = (datetime.now() - date).days
            
            if days_ago == 0:
                return "Today"
            elif days_ago == 1:
                return "Yesterday"
            elif days_ago < 7:
                return f"{days_ago} days ago"
            elif days_ago < 30:
                weeks = days_ago // 7
                return f"{weeks} week{'s' if weeks > 1 else ''} ago"
            elif days_ago < 365:
                months = days_ago // 30
                return f"{months} month{'s' if months > 1 else ''} ago"
            else:
                years = days_ago // 365
                return f"{years} year{'s' if years > 1 else ''} ago"
        except:
            return upload_date
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 12) -> List[Dict]:
        """Get video recommendations for a user."""
        if not hasattr(self, 'items_df'):
            return []
        
        # Get user preferences
        liked_videos = self.get_session_value('liked_items', set())
        disliked_videos = self.get_session_value('disliked_items', set())
        selected_category = self.get_session_value('selected_category', 'All')
        duration_filter = self.get_session_value('duration_filter', 'Any')
        quality_filter = self.get_session_value('quality_filter', 'Any')
        sort_by = self.get_session_value('sort_by', 'Recommended')
        num_recs = self.get_session_value('num_recommendations', num_recommendations)
        
        # Filter videos
        df = self.items_df.copy()
        
        # Remove already interacted videos
        all_interacted = liked_videos.union(disliked_videos)
        if all_interacted:
            df = df[~df['video_id'].isin(all_interacted)]
        
        # Apply category filter
        if selected_category != 'All':
            df = df[df['category'] == selected_category]
        
        # Apply duration filter
        if duration_filter != 'Any':
            if duration_filter == "Short (< 4 min)":
                df = df[df['duration_seconds'] < 240]
            elif duration_filter == "Medium (4-20 min)":
                df = df[(df['duration_seconds'] >= 240) & (df['duration_seconds'] <= 1200)]
            elif duration_filter == "Long (> 20 min)":
                df = df[df['duration_seconds'] > 1200]
        
        # Apply quality filter
        if quality_filter != 'Any':
            if quality_filter == "High Quality (>0.7)":
                df = df[df['quality_score'] > 0.7]
            elif quality_filter == "Medium Quality (>0.4)":
                df = df[df['quality_score'] > 0.4]
        
        # Apply sorting
        if sort_by == "Most Views":
            df = df.sort_values('views', ascending=False)
        elif sort_by == "Most Recent":
            df['upload_date'] = pd.to_datetime(df['upload_date'])
            df = df.sort_values('upload_date', ascending=False)
        elif sort_by == "Most Liked":
            df = df.sort_values('likes', ascending=False)
        elif sort_by == "Trending":
            # Trending score based on recent views and engagement
            df['trending_score'] = (
                df['views'] * 0.4 +
                df['likes'] * 10 * 0.3 +
                df['engagement_rate'] * 100000 * 0.3
            )
            df = df.sort_values('trending_score', ascending=False)
        else:  # Recommended
            if liked_videos:
                df = self._get_content_based_recommendations(df, liked_videos)
            else:
                # No preferences, show trending
                df['trending_score'] = (
                    df['views'] * 0.5 +
                    df['likes'] * 10 * 0.5
                )
                df = df.sort_values('trending_score', ascending=False)
        
        # Return top recommendations
        recommendations = df.head(num_recs).to_dict('records')
        return recommendations
    
    def _get_content_based_recommendations(self, df: pd.DataFrame, liked_videos: set) -> pd.DataFrame:
        """Get content-based recommendations based on liked videos."""
        if df.empty:
            return df
        
        # Get liked videos features
        liked_df = self.items_df[self.items_df['video_id'].isin(liked_videos)]
        
        if not liked_df.empty:
            # Calculate average features of liked videos
            feature_cols = ['quality_score', 'engagement_rate', 'educational_value', 'entertainment_value']
            avg_features = liked_df[feature_cols].mean()
            
            # Get preferred categories
            preferred_categories = liked_df['category'].value_counts()
            
            # Calculate similarity score
            for col in feature_cols:
                df[f'{col}_diff'] = abs(df[col] - avg_features[col])
            
            # Category preference boost
            df['category_boost'] = df['category'].map(
                lambda x: preferred_categories.get(x, 0) / len(liked_videos)
            ).fillna(0)
            
            # Overall similarity score
            df['similarity_score'] = (
                (1 - df['quality_score_diff']) * 0.25 +
                (1 - df['engagement_rate_diff']) * 0.25 +
                (1 - df['educational_value_diff']) * 0.15 +
                (1 - df['entertainment_value_diff']) * 0.15 +
                df['category_boost'] * 0.2
            )
            
            # Sort by similarity
            df = df.sort_values('similarity_score', ascending=False)
        
        return df
    
    def _get_trending_videos(self, num_videos: int = 12) -> List[Dict]:
        """Get trending videos."""
        if not hasattr(self, 'items_df'):
            return []
        
        df = self.items_df.copy()
        
        # Calculate trending score
        df['trending_score'] = (
            df['views'] * 0.4 +
            df['likes'] * 10 * 0.3 +
            df['engagement_rate'] * 100000 * 0.3
        )
        
        return df.nlargest(num_videos, 'trending_score').to_dict('records')


def generate_sample_data(file_path: str, num_videos: int = 1000):
    """Generate sample video data file."""
    frontend = YouTubeFrontend()
    
    # Override the number of videos to generate
    import random
    from datetime import datetime, timedelta
    import pandas as pd
    
    # Sample channels and categories
    channels = [
        'TechReviews', 'CookingMaster', 'FitnessGuru', 'MusicVibes', 'GameZone',
        'TravelWorld', 'ScienceHub', 'ArtStudio', 'NewsDaily', 'ComedyCentral',
        'EducationPlus', 'DIYProjects', 'FashionTrends', 'AutoReviews', 'PetLovers',
        'BusinessInsights', 'HealthTips', 'MovieReviews', 'SportsCentral', 'TechTalks'
    ]
    
    categories = [
        'Technology', 'Cooking', 'Fitness', 'Music', 'Gaming',
        'Travel', 'Science', 'Art', 'News', 'Comedy',
        'Education', 'DIY', 'Fashion', 'Automotive', 'Pets',
        'Business', 'Health', 'Movies', 'Sports', 'Lifestyle'
    ]
    
    # Sample video title templates
    title_templates = [
        "How to {action} {object}",
        "Top 10 {adjective} {category}",
        "Ultimate Guide to {topic}",
        "Amazing {adjective} {object}",
        "Why {topic} is {adjective}",
        "Best {object} for {purpose}",
        "Incredible {adjective} {topic}",
        "{number} {adjective} {category} Tips"
    ]
    
    actions = ["build", "create", "fix", "improve", "master", "learn", "discover"]
    objects = ["gadgets", "recipes", "workouts", "songs", "games", "places", "experiments"]
    adjectives = ["amazing", "incredible", "ultimate", "best", "awesome", "mind-blowing", "surprising"]
    topics = ["technology", "cooking", "fitness", "music", "gaming", "travel", "science"]
    purposes = ["beginners", "professionals", "home", "work", "travel", "kids", "adults"]
    
    # Generate videos
    videos_data = []
    for i in range(num_videos):
        video_id = f"video_{i:04d}"
        channel = random.choice(channels)
        category = random.choice(categories)
        
        # Generate title
        template = random.choice(title_templates)
        title = template.format(
            action=random.choice(actions),
            object=random.choice(objects),
            adjective=random.choice(adjectives),
            category=category.lower(),
            topic=random.choice(topics),
            purpose=random.choice(purposes),
            number=random.randint(3, 20)
        )
        
        # Video metrics
        views = random.randint(100, 10000000)
        likes = int(views * random.uniform(0.01, 0.15))
        duration_seconds = random.randint(30, 3600)  # 30 sec to 1 hour
        
        # Upload date
        upload_date = datetime.now() - timedelta(days=random.randint(0, 1095))  # Up to 3 years ago
        
        # Video features
        quality_score = random.random()
        engagement_rate = random.random()
        educational_value = random.random()
        entertainment_value = random.random()
        
        videos_data.append({
            'video_id': video_id,
            'title': title,
            'channel_name': channel,
            'category': category,
            'duration_seconds': duration_seconds,
            'views': views,
            'likes': likes,
            'upload_date': upload_date.strftime('%Y-%m-%d'),
            'quality_score': quality_score,
            'engagement_rate': engagement_rate,
            'educational_value': educational_value,
            'entertainment_value': entertainment_value,
            'thumbnail_url': f"https://img.youtube.com/vi/sample_{i}/maxresdefault.jpg",
            'description': f"An amazing {category.lower()} video from {channel}. {title}"
        })
    
    videos_df = pd.DataFrame(videos_data)
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to CSV
    videos_df.to_csv(file_path, index=False)
    print(f"Generated {num_videos} sample videos at {file_path}") 