"""
Frontend Manager for CoreRec Demo Frontends

This module provides a unified interface for managing and launching
different platform demonstration frontends.
"""

import streamlit as st
import importlib
import os
from typing import Dict, List, Type, Any, Optional
from .base_frontend import BaseFrontend


class FrontendManager:
    """
    Manages different demo frontend platforms and provides a unified interface
    for launching and switching between them.
    """

    def __init__(self, recommendation_engine: Any = None, data_base_path: str = None):
        """
        Initialize the Frontend Manager.

        Args:
            recommendation_engine: CoreRec recommendation engine instance
            data_base_path: Base path for data files
        """
        self.recommendation_engine = recommendation_engine
        self.data_base_path = data_base_path or "datasets"

        # Registry of available platforms
        self.platforms = {
            "spotify": {
                "name": "Spotify Music Streaming",
                "description": "Music recommendation platform similar to Spotify",
                "icon": "🎵",
                "class_name": "SpotifyFrontend",
                "module_path": "corerec.demo_frontends.platforms.spotify_frontend",
                "sample_data": "music_tracks.csv",
            },
            "youtube": {
                "name": "YouTube Video Platform",
                "description": "Video recommendation platform similar to YouTube",
                "icon": "📺",
                "class_name": "YouTubeFrontend",
                "module_path": "corerec.demo_frontends.platforms.youtube_frontend",
                "sample_data": "videos.csv",
            },
            "netflix": {
                "name": "Netflix Streaming",
                "description": "Movie and TV show recommendation platform",
                "icon": "🎬",
                "class_name": "NetflixFrontend",
                "module_path": "corerec.demo_frontends.platforms.netflix_frontend",
                "sample_data": "movies_shows.csv",
            },
        }

    def get_available_platforms(self) -> Dict[str, Dict]:
        """Get dictionary of available platforms."""
        return self.platforms

    def load_platform_frontend(self, platform_key: str) -> BaseFrontend:
        """
        Dynamically load and instantiate a platform frontend.

        Args:
            platform_key: Key identifying the platform (e.g., 'spotify', 'youtube')

        Returns:
            Instance of the platform frontend class
        """
        if platform_key not in self.platforms:
            raise ValueError(
                f"Platform '{platform_key}' not found. Available platforms: {list(self.platforms.keys())}"
            )

        platform_config = self.platforms[platform_key]

        try:
            # Dynamically import the module
            module = importlib.import_module(platform_config["module_path"])

            # Get the class
            frontend_class = getattr(module, platform_config["class_name"])

            # Prepare data path
            data_path = os.path.join(self.data_base_path, platform_config["sample_data"])

            # Instantiate the frontend
            return frontend_class(
                recommendation_engine=self.recommendation_engine, data_path=data_path
            )

        except ImportError as e:
            st.error(f"Failed to import {platform_config['class_name']}: {str(e)}")
            raise
        except AttributeError as e:
            st.error(f"Class {platform_config['class_name']} not found in module: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error initializing {platform_config['name']}: {str(e)}")
            raise

    def render_platform_selector(self) -> Optional[str]:
        """
        Render a platform selection interface.

        Returns:
            Selected platform key or None
        """
        st.markdown(
            """
        <div style="text-align: center; padding: 2rem;">
            <h1>🚀 CoreRec Demo Frontends</h1>
            <h3>Choose a platform to demonstrate your recommendation model</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Create columns for platform cards
        cols = st.columns(3)
        selected_platform = None

        for i, (platform_key, platform_info) in enumerate(self.platforms.items()):
            col_idx = i % 3
            with cols[col_idx]:
                # Create a card for each platform
                card_html = f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    color: white;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    transition: transform 0.2s ease;
                ">
                    <h2>{platform_info['icon']}</h2>
                    <h3>{platform_info['name']}</h3>
                    <p>{platform_info['description']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                if st.button(f"Launch {platform_info['name']}", key=f"launch_{platform_key}"):
                    selected_platform = platform_key

        return selected_platform

    def run_demo_selector(self):
        """Run the main demo platform selector."""
        st.set_page_config(page_title="CoreRec Demo Frontends", page_icon="🚀", layout="wide")

        # Custom CSS for the selector
        st.markdown(
            """
        <style>
        .main > div {
            padding: 2rem;
        }
        
        .platform-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        
        .platform-card:hover {
            transform: translateY(-5px);
        }
        
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: bold;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Initialize session state
        if "selected_platform" not in st.session_state:
            st.session_state.selected_platform = None

        # Show platform selector or run selected platform
        if st.session_state.selected_platform is None:
            selected = self.render_platform_selector()
            if selected:
                st.session_state.selected_platform = selected
                st.rerun()
        else:
            # Show back button
            if st.button("← Back to Platform Selection"):
                st.session_state.selected_platform = None
                st.rerun()

            # Load and run the selected platform
            try:
                frontend = self.load_platform_frontend(st.session_state.selected_platform)
                frontend.run()
            except Exception as e:
                st.error(f"Error running platform: {str(e)}")
                if st.button("Try Again"):
                    st.session_state.selected_platform = None
                    st.rerun()

    def generate_sample_data(self, platform_key: str, num_items: int = 1000):
        """
        Generate sample data for a platform if it doesn't exist.

        Args:
            platform_key: Platform to generate data for
            num_items: Number of sample items to generate
        """
        if platform_key not in self.platforms:
            raise ValueError(f"Platform '{platform_key}' not found")

        platform_config = self.platforms[platform_key]
        data_path = os.path.join(self.data_base_path, platform_config["sample_data"])

        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Import the platform-specific data generator
        try:
            module = importlib.import_module(platform_config["module_path"])
            if hasattr(module, "generate_sample_data"):
                module.generate_sample_data(data_path, num_items)
                return True
        except Exception as e:
            st.warning(f"Could not generate sample data for {platform_key}: {str(e)}")

        return False

    def run_platform(self, platform_name: str):
        """
        Run a specific platform frontend.

        Args:
            platform_name: Name of the platform to run ('spotify', 'youtube', 'netflix')
        """
        try:
            frontend = self.load_platform_frontend(platform_name)
            frontend.run()
        except Exception as e:
            st.error(f"Error running {platform_name} frontend: {str(e)}")
            raise

    def render_platform_selection(self):
        """
        Render platform selection interface and handle platform selection.
        """
        selected = self.render_platform_selector()
        if selected:
            st.session_state["selected_platform"] = selected
            st.rerun()


def quick_launch(platform_name: str, recommendation_engine: Any = None, data_path: str = None):
    """
    Quick launch function for directly starting a specific platform demo.

    Args:
        platform_name: Name of the platform to launch ('spotify', 'youtube', etc.)
        recommendation_engine: CoreRec recommendation engine instance
        data_path: Path to data files

    Example:
        ```python
        from corerec.demo_frontends import quick_launch
        from corerec.engines.hybrid import HybridEngine

        # Create your recommendation engine
        engine = HybridEngine()

        # Launch Spotify demo
        quick_launch('spotify', engine)
        ```
    """
    manager = FrontendManager(recommendation_engine, data_path)

    try:
        frontend = manager.load_platform_frontend(platform_name)
        frontend.run()
    except Exception as e:
        st.error(f"Error launching {platform_name} demo: {str(e)}")
        st.info("Available platforms: " + ", ".join(manager.get_available_platforms().keys()))
