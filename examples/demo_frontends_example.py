#!/usr/bin/env python3
"""
CoreRec Demo Frontends Usage Example

This script demonstrates how to use the CoreRec demo frontends
to showcase recommendation models with beautiful, platform-specific interfaces.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def example_basic_usage():
    """Example: Basic usage with default sample data."""
    print("=" * 60)
    print("Example 1: Basic Usage with Sample Data")
    print("=" * 60)
    
    from corerec.demo_frontends import quick_launch
    
    # Launch Spotify demo with generated sample data
    print("Launching Spotify demo...")
    print("Run: streamlit run corerec/demo_frontends/demo_app.py -- --platform spotify")
    
    # You can also launch directly:
    # quick_launch('spotify')

def example_with_custom_engine():
    """Example: Using demo frontends with a custom recommendation engine."""
    print("=" * 60)
    print("Example 2: With Custom Recommendation Engine")
    print("=" * 60)
    
    # This would be your custom recommendation engine
    class CustomRecommendationEngine:
        def __init__(self):
            self.model_name = "Custom Collaborative Filtering"
        
        def get_recommendations(self, user_id, num_recommendations=10):
            # Your recommendation logic here
            return []
    
    from corerec.demo_frontends import SpotifyFrontend
    
    # Create your custom engine
    engine = CustomRecommendationEngine()
    
    # Initialize frontend with your engine
    frontend = SpotifyFrontend(recommendation_engine=engine)
    
    print("Frontend created with custom engine!")
    print("To run: frontend.run() in a Streamlit context")

def example_with_custom_data():
    """Example: Using custom data files."""
    print("=" * 60)
    print("Example 3: With Custom Data Files")
    print("=" * 60)
    
    from corerec.demo_frontends import YouTubeFrontend
    import pandas as pd
    
    # Create sample custom data
    custom_data = {
        'video_id': ['vid_001', 'vid_002', 'vid_003'],
        'title': ['Amazing AI Tutorial', 'Machine Learning Basics', 'Deep Learning Demo'],
        'channel_name': ['TechChannel', 'MLHub', 'AILearning'],
        'category': ['Technology', 'Education', 'Technology'],
        'duration_seconds': [600, 1200, 900],
        'views': [50000, 25000, 75000],
        'likes': [2500, 1200, 3800],
        'upload_date': ['2024-01-15', '2024-02-01', '2024-03-10'],
        'quality_score': [0.9, 0.8, 0.95],
        'engagement_rate': [0.7, 0.6, 0.8],
        'educational_value': [0.9, 0.95, 0.85],
        'entertainment_value': [0.7, 0.6, 0.8],
        'thumbnail_url': ['url1', 'url2', 'url3'],
        'description': ['An amazing tutorial', 'Learn ML basics', 'Deep learning made easy']
    }
    
    # Save custom data
    df = pd.DataFrame(custom_data)
    os.makedirs('custom_data', exist_ok=True)
    df.to_csv('custom_data/my_videos.csv', index=False)
    
    # Use with frontend
    frontend = YouTubeFrontend(data_path='custom_data/my_videos.csv')
    
    print("Custom data created at: custom_data/my_videos.csv")
    print("Frontend initialized with custom data!")

def example_platform_manager():
    """Example: Using the FrontendManager for multiple platforms."""
    print("=" * 60)
    print("Example 4: Using FrontendManager")
    print("=" * 60)
    
    from corerec.demo_frontends import FrontendManager
    
    # Initialize manager
    manager = FrontendManager()
    
    # Get available platforms
    platforms = manager.get_available_platforms()
    print("Available platforms:")
    for key, info in platforms.items():
        print(f"  - {key}: {info['name']} ({info['description']})")
    
    # Load a specific platform
    try:
        spotify_frontend = manager.load_platform_frontend('spotify')
        print("\nSpotify frontend loaded successfully!")
    except Exception as e:
        print(f"Error loading Spotify frontend: {e}")

def example_sample_data_generation():
    """Example: Generating sample data for different platforms."""
    print("=" * 60)
    print("Example 5: Sample Data Generation")
    print("=" * 60)
    
    # Generate sample data for all platforms
    platforms = {
        'spotify': 'corerec.demo_frontends.platforms.spotify_frontend',
        'youtube': 'corerec.demo_frontends.platforms.youtube_frontend',
        'netflix': 'corerec.demo_frontends.platforms.netflix_frontend'
    }
    
    for platform, module_path in platforms.items():
        try:
            # Import the module dynamically
            module = __import__(module_path, fromlist=['generate_sample_data'])
            
            # Generate sample data
            output_file = f'sample_data/{platform}_demo.csv'
            os.makedirs('sample_data', exist_ok=True)
            
            module.generate_sample_data(output_file, 100)
            print(f"✅ Generated {platform} sample data: {output_file}")
            
        except Exception as e:
            print(f"❌ Error generating {platform} data: {e}")

def main():
    """Run all examples."""
    print("🎯 CoreRec Demo Frontends - Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    print()
    
    example_with_custom_engine()
    print()
    
    example_with_custom_data()
    print()
    
    example_platform_manager()
    print()
    
    example_sample_data_generation()
    print()
    
    print("=" * 60)
    print("🚀 Ready to launch your demo!")
    print("Run: streamlit run corerec/demo_frontends/demo_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 