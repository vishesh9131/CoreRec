#!/usr/bin/env python3
"""
Test script for CoreRec IMShow Example 1 - Simple Music Recommender
"""

import random
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the CoreRec IMShow connector
from corerec.imshow import connector


def simple_music_recommender(user_id, num_items=10):
    """
    Simple music recommendation function.
    Returns a list of music recommendations with required fields.
    """
    print(f"🎵 Generating {num_items} music recommendations for user: {user_id}")

    # Sample data for realistic recommendations
    artists = [
        "Taylor Swift",
        "Drake",
        "The Weeknd",
        "Billie Eilish",
        "Ed Sheeran",
        "Ariana Grande",
        "Post Malone",
        "Dua Lipa",
        "Harry Styles",
        "Olivia Rodrigo",
    ]
    genres = ["Pop", "Hip-Hop", "Electronic", "Indie", "Rock", "R&B", "Country"]

    recommendations = []
    for i in range(num_items):
        recommendations.append(
            {
                "id": f"song_{user_id}_{i+1}",
                "title": f"Amazing Song {i+1}",
                "artist": random.choice(artists),
                "album": f"Great Album {i+1}",
                "genre": random.choice(genres),
                "duration": f"{random.randint(180, 300)}s",
                "popularity": random.randint(60, 100),
            }
        )

    return recommendations


def main():
    print("🎯 CoreRec IMShow Example 1: Simple Music Recommender")
    print("=" * 60)
    print("This example demonstrates how to plug a simple recommendation")
    print("function into a beautiful Spotify-style frontend interface.")
    print()

    # Create the connector
    demo = connector(
        predict_function=simple_music_recommender,
        frontend="spotify",
        title="My Music Recommendation System",
        description="Personalized music recommendations using collaborative filtering",
        debug=True,  # Enable debug mode to see what's happening
    )

    print("🚀 Starting Spotify-style music demo...")
    print("📱 This will open your browser with a beautiful interface!")
    print("🎵 Click on the play buttons to interact with recommendations")
    print()
    print("💡 To stop the demo, press Ctrl+C in this terminal")
    print()

    try:
        # Run the demo (this will open browser and start servers)
        demo.run()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        demo.stop()
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure all dependencies are installed and ports are available")


if __name__ == "__main__":
    main()
