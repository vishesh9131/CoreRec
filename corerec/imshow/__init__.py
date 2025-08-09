"""
CoreRec IMShow - Instant Demo Frontend Connector

A simple plug-and-play system for connecting your recommendation models 
to beautiful, pre-built frontend interfaces.

Usage:
    import corerec.imshow as ii
    
    # Your recommendation function
    def my_recommender(user_id, num_items=10):
        # Your recommendation logic here
        return recommendations
    
    # Plug into a frontend
    connector = ii.connector(
        predict_function=my_recommender,
        frontend="spotify",  # or "youtube", "netflix"
        title="My Music Recommender",
        port=8080
    )
    
    # Launch the demo
    connector.run()

Available frontends:
    - spotify: Music streaming platform with dark theme
    - youtube: Video platform with red theme  
    - netflix: Movie and TV show platform with dark theme

Quick Start:
    # Simple example
    def recommend(user_id, num_items=10):
        return [{"id": i, "title": f"Item {i}"} for i in range(num_items)]
    
    demo = ii.connector(recommend, frontend="spotify")
    demo.run()
"""

from .connector import Connector, connector
from .frontends import available_frontends, get_frontend_info
from .utils import create_sample_data, format_recommendations

# Version info
__version__ = "1.0.0"
__author__ = "CoreRec Team"

# Public API
__all__ = [
    "connector",
    "Connector", 
    "available_frontends",
    "get_frontend_info",
    "create_sample_data"
]

def list_frontends():
    """
    List all available frontends with their descriptions.
    
    Returns:
        Dictionary of available frontends
    """
    frontends = available_frontends()
    
    print("🎨 Available IMShow Frontends:")
    print("=" * 40)
    
    for key, info in frontends.items():
        print(f"\n🔹 {key}")
        print(f"   Name: {info['name']}")
        print(f"   Use case: {info['use_case']}")
        print(f"   Theme: {info['theme']}")
        print(f"   Required fields: {', '.join(info['required_fields'])}")
    
    print(f"\n💡 Usage: connector(my_function, frontend='frontend_name')")
    
    return frontends

def quick_demo(frontend: str = "spotify", port: int = 8080):
    """
    Launch a quick demo with sample data.
    
    Args:
        frontend: Frontend type to demo
        port: Port to run on
    """
    def sample_recommender(user_id, num_items=12):
        """Sample recommendation function."""
        return create_sample_data(frontend, num_items)
    
    demo = connector(
        predict_function=sample_recommender,
        frontend=frontend,
        title=f"CoreRec IMShow - {frontend.title()} Demo",
        description="This is a sample demo with mock data. Replace with your own recommendation function!",
        port=port
    )
    
    print(f"🚀 Launching quick demo for {frontend}...")
    print("💡 This uses sample data. Create your own with connector(your_function, frontend)")
    
    demo.run()

def validate_function(func):
    """
    Validate that a function can be used with IMShow.
    
    Args:
        func: Function to validate
        
    Returns:
        True if valid, raises exception if not
    """
    from .utils import validate_prediction_function
    
    try:
        validate_prediction_function(func)
        print(f"✅ Function '{func.__name__}' is compatible with IMShow")
        return True
    except Exception as e:
        print(f"❌ Function validation failed: {e}")
        raise

# Convenience functions for each frontend
def spotify_demo(predict_function, title: str = "Music Recommender", **kwargs):
    """Create a Spotify-themed demo."""
    return connector(
        predict_function=predict_function,
        frontend="spotify",
        title=title,
        description="Music recommendations with Spotify-style interface",
        **kwargs
    )

def youtube_demo(predict_function, title: str = "Video Recommender", **kwargs):
    """Create a YouTube-themed demo."""
    return connector(
        predict_function=predict_function,
        frontend="youtube", 
        title=title,
        description="Video recommendations with YouTube-style interface",
        **kwargs
    )

def netflix_demo(predict_function, title: str = "Movie Recommender", **kwargs):
    """Create a Netflix-themed demo."""
    return connector(
        predict_function=predict_function,
        frontend="netflix",
        title=title,
        description="Movie/TV recommendations with Netflix-style interface", 
        **kwargs
    )

# Add convenience functions to __all__
__all__.extend([
    "list_frontends",
    "quick_demo", 
    "validate_function",
    "spotify_demo",
    "youtube_demo", 
    "netflix_demo"
])

# Show welcome message on import
def _show_welcome():
    """Show welcome message when module is imported."""
    print("🎯 CoreRec IMShow loaded!")
    print("💡 Quick start: ii.connector(your_function, frontend='spotify').run()")
    print("📚 List frontends: ii.list_frontends()")
    print("🚀 Quick demo: ii.quick_demo('spotify')")

# Only show welcome in interactive mode
try:
    # Check if we're in an interactive session
    import sys
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        _show_welcome()
except:
    pass 