"""
Base Frontend Class for CoreRec Demo Frontends

This class provides the foundation for all platform-specific frontends,
defining the common interface and shared functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
from datetime import datetime


class BaseFrontend(ABC):
    """
    Abstract base class for all demo frontends.
    
    This class defines the common interface that all platform-specific
    frontends must implement, ensuring consistency across different
    demonstration platforms.
    """
    
    def __init__(self, 
                 platform_name: str,
                 recommendation_engine: Any = None,
                 data_path: str = None,
                 theme_config: Dict = None):
        """
        Initialize the base frontend.
        
        Args:
            platform_name: Name of the platform (e.g., "Spotify", "YouTube")
            recommendation_engine: CoreRec recommendation engine instance
            data_path: Path to the data files
            theme_config: Custom theme configuration
        """
        self.platform_name = platform_name
        self.recommendation_engine = recommendation_engine
        self.data_path = data_path
        self.theme_config = theme_config or self._get_default_theme()
        
        # Initialize session state keys specific to this platform
        self.session_prefix = f"{platform_name.lower()}_"
        self._initialize_session_state()
        
        # Setup logging
        self.logger = logging.getLogger(f"CoreRec.{platform_name}")
        
    def _get_default_theme(self) -> Dict:
        """Get default theme configuration."""
        return {
            'primary_color': '#1DB954',
            'background_color': '#191414',
            'text_color': '#FFFFFF',
            'secondary_color': '#1ED760',
            'accent_color': '#FF6B6B'
        }
    
    def _initialize_session_state(self):
        """Initialize platform-specific session state variables."""
        session_vars = {
            f'{self.session_prefix}initialized': False,
            f'{self.session_prefix}user_id': 1,
            f'{self.session_prefix}user_preferences': {},
            f'{self.session_prefix}liked_items': set(),
            f'{self.session_prefix}disliked_items': set(),
            f'{self.session_prefix}interaction_history': [],
            f'{self.session_prefix}current_recommendations': [],
            f'{self.session_prefix}page_state': 'home'
        }
        
        for key, default_value in session_vars.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def get_session_key(self, key: str) -> str:
        """Get the full session state key with platform prefix."""
        return f"{self.session_prefix}{key}"
    
    def get_session_value(self, key: str, default=None):
        """Get value from session state with platform prefix."""
        return st.session_state.get(self.get_session_key(key), default)
    
    def set_session_value(self, key: str, value):
        """Set value in session state with platform prefix."""
        st.session_state[self.get_session_key(key)] = value
    
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load platform-specific data.
        
        Returns:
            Tuple of (items_df, interactions_df)
            items_df: DataFrame containing item information
            interactions_df: DataFrame containing user-item interactions (optional)
        """
        pass
    
    @abstractmethod
    def render_header(self):
        """Render the platform-specific header/navigation."""
        pass
    
    @abstractmethod
    def render_sidebar(self):
        """Render the platform-specific sidebar with controls."""
        pass
    
    @abstractmethod
    def render_main_content(self):
        """Render the main content area."""
        pass
    
    @abstractmethod
    def render_item_card(self, item: Dict, index: int = 0) -> bool:
        """
        Render an individual item card.
        
        Args:
            item: Dictionary containing item information
            index: Index for unique widget keys
            
        Returns:
            Boolean indicating if user interacted with the item
        """
        pass
    
    @abstractmethod
    def get_recommendations(self, user_id: int, num_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID to get recommendations for
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of item dictionaries
        """
        pass
    
    def apply_custom_css(self):
        """Apply platform-specific CSS styling."""
        st.markdown(f"""
        <style>
        .main-header {{
            background: linear-gradient(135deg, {self.theme_config['primary_color']}, {self.theme_config['secondary_color']});
            color: {self.theme_config['text_color']};
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        
        .item-card {{
            background-color: {self.theme_config['background_color']};
            color: {self.theme_config['text_color']};
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s ease;
        }}
        
        .item-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }}
        
        .platform-title {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .platform-subtitle {{
            font-size: 1.2rem;
            opacity: 0.8;
        }}
        
        .interaction-button {{
            background-color: {self.theme_config['accent_color']};
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1.5rem;
            margin: 0.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        
        .interaction-button:hover {{
            transform: scale(1.05);
            opacity: 0.9;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def record_interaction(self, item_id: Any, interaction_type: str, rating: float = None):
        """Record user interaction with an item."""
        interaction = {
            'timestamp': datetime.now(),
            'item_id': item_id,
            'interaction_type': interaction_type,  # 'like', 'dislike', 'view', 'click'
            'rating': rating
        }
        
        # Update session state
        interactions = self.get_session_value('interaction_history', [])
        interactions.append(interaction)
        self.set_session_value('interaction_history', interactions)
        
        # Update liked/disliked sets
        if interaction_type == 'like':
            liked = self.get_session_value('liked_items', set())
            liked.add(item_id)
            self.set_session_value('liked_items', liked)
            
            # Remove from disliked if present
            disliked = self.get_session_value('disliked_items', set())
            disliked.discard(item_id)
            self.set_session_value('disliked_items', disliked)
            
        elif interaction_type == 'dislike':
            disliked = self.get_session_value('disliked_items', set())
            disliked.add(item_id)
            self.set_session_value('disliked_items', disliked)
            
            # Remove from liked if present
            liked = self.get_session_value('liked_items', set())
            liked.discard(item_id)
            self.set_session_value('liked_items', liked)
    
    def get_user_feedback_summary(self) -> Dict:
        """Get summary of user feedback and interactions."""
        return {
            'total_interactions': len(self.get_session_value('interaction_history', [])),
            'liked_items': len(self.get_session_value('liked_items', set())),
            'disliked_items': len(self.get_session_value('disliked_items', set())),
            'current_user_id': self.get_session_value('user_id', 1)
        }
    
    def render_feedback_summary(self):
        """Render user feedback summary in sidebar."""
        summary = self.get_user_feedback_summary()
        
        st.sidebar.markdown("### 📊 Your Activity")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("❤️ Liked", summary['liked_items'])
        with col2:
            st.metric("👎 Disliked", summary['disliked_items'])
        
        st.sidebar.metric("🔄 Total Interactions", summary['total_interactions'])
    
    def render_user_controls(self):
        """Render user controls in sidebar."""
        st.sidebar.markdown("### 👤 User Settings")
        
        # User ID selector
        current_user = self.get_session_value('user_id', 1)
        new_user = st.sidebar.number_input("User ID", min_value=1, max_value=1000, value=current_user)
        
        if new_user != current_user:
            self.set_session_value('user_id', new_user)
            # Reset user-specific data
            self.set_session_value('liked_items', set())
            self.set_session_value('disliked_items', set())
            self.set_session_value('interaction_history', [])
            st.rerun()
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History"):
            self.set_session_value('liked_items', set())
            self.set_session_value('disliked_items', set())
            self.set_session_value('interaction_history', [])
            st.rerun()
    
    def run(self):
        """Main method to run the frontend application."""
        # Configure Streamlit page
        st.set_page_config(
            page_title=f"{self.platform_name} Demo - Powered by CoreRec",
            page_icon="🎵" if self.platform_name == "Spotify" else "📺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom styling
        self.apply_custom_css()
        
        # Load data if not already loaded
        if not self.get_session_value('initialized'):
            with st.spinner(f"Loading {self.platform_name} demo..."):
                try:
                    self.items_df, self.interactions_df = self.load_data()
                    self.set_session_value('initialized', True)
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.stop()
        
        # Render components
        self.render_header()
        
        # Sidebar
        with st.sidebar:
            self.render_user_controls()
            self.render_feedback_summary()
            self.render_sidebar()
        
        # Main content
        self.render_main_content() 