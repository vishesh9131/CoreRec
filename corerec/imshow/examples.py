"""
Comprehensive examples showing how to use CoreRec IMShow.

Run these examples to see how different types of recommendation systems
can be quickly plugged into beautiful frontend interfaces.
"""

import random
import time
from typing import List, Dict, Any

# Import the IMShow connector
try:
    from . import connector, available_frontends
except ImportError:
    # For standalone execution
    import sys

    sys.path.append("..")
    from corerec.imshow import connector, available_frontends


def example_1_simple_music_recommender():
    """
    Example 1: Simple music recommendation function.

    This is the most basic example - a function that returns
    a list of music recommendations.
    """
    print("🎵 Example 1: Simple Music Recommender")
    print("=" * 50)

    def my_music_recommender(user_id, num_items=10):
        """Simple music recommendation function."""
        artists = [
            "Taylor Swift",
            "Drake",
            "The Weeknd",
            "Billie Eilish",
            "Ed Sheeran"]
        genres = ["Pop", "Hip-Hop", "Electronic", "Indie", "Rock"]

        recommendations = []
        for i in range(num_items):
            recommendations.append(
                {
                    "id": f"song_{i + 1}",
                    "title": f"Amazing Song {i + 1}",
                    "artist": random.choice(artists),
                    "album": f"Great Album {i + 1}",
                    "genre": random.choice(genres),
                    "duration": f"{random.randint(180, 300)}s",
                }
            )

        return recommendations

    # Create and run the demo
    demo = connector(
        predict_function=my_music_recommender,
        frontend="spotify",
        title="My Music Recommendation System",
        description="Personalized music recommendations using collaborative filtering",
    )

    print("🚀 Starting Spotify-style music demo...")
    demo.run()


def example_2_video_recommender_with_different_params():
    """
    Example 2: Video recommender with different parameter names.

    Shows how the system automatically maps different parameter names
    like 'k' instead of 'num_items'.
    """
    print("📺 Example 2: Video Recommender (Different Parameters)")
    print("=" * 50)

    def my_video_recommender(user_id, k=12):
        """Video recommendation function using 'k' parameter."""
        channels = [
            "TechReview",
            "CookingMaster",
            "GameZone",
            "TravelVlog",
            "ScienceExplained"]
        categories = ["Technology", "Food", "Gaming", "Travel", "Education"]

        videos = []
        for i in range(k):
            views = random.randint(1000, 5000000)
            duration_min = random.randint(2, 30)
            duration_sec = random.randint(0, 59)

            videos.append(
                {
                    "id": f"video_{i + 1}",
                    "title": f"Incredible Video {i + 1} - Must Watch!",
                    "channel": random.choice(channels),
                    "category": random.choice(categories),
                    "views": f"{views:,}",
                    "duration": f"{duration_min}:{duration_sec:02d}",
                    "uploaded": f"{random.randint(1, 30)} days ago",
                }
            )

        return videos

    # Create YouTube-style demo
    demo = connector(
        predict_function=my_video_recommender,
        frontend="youtube",
        title="Smart Video Recommendation Engine",
        description="AI-powered video recommendations based on viewing history",
        port=8081,  # Use different port
    )

    print("🚀 Starting YouTube-style video demo...")
    demo.run()


def example_3_netflix_movie_recommender():
    """
    Example 3: Netflix-style movie/TV show recommender.

    Shows how to create content recommendations for streaming platforms.
    """
    print("🎬 Example 3: Netflix Movie & TV Recommender")
    print("=" * 50)

    def movie_tv_recommender(user_id, num_recommendations=15):
        """Movie and TV show recommendation function."""
        genres = [
            "Action",
            "Comedy",
            "Drama",
            "Thriller",
            "Romance",
            "Sci-Fi",
            "Horror",
            "Documentary",
        ]
        content_types = ["Movie", "TV Show", "Limited Series", "Documentary"]

        recommendations = []
        for i in range(num_recommendations):
            content_type = random.choice(content_types)

            recommendation = {
                "id": f"content_{i + 1}",
                "title": f"Amazing {content_type} {i + 1}",
                "genre": random.choice(genres),
                "year": random.randint(2018, 2024),
                "rating": f"{random.uniform(7.5, 9.5):.1f}",
                "type": content_type,
            }

            # Add type-specific fields
            if content_type == "Movie":
                recommendation["duration"] = f"{random.randint(90, 180)} min"
            else:
                recommendation["duration"] = f"{random.randint(1, 8)} Seasons"

            recommendations.append(recommendation)

        return recommendations

    # Create Netflix-style demo
    demo = connector(
        predict_function=movie_tv_recommender,
        frontend="netflix",
        title="StreamSmart Recommendation Engine",
        description="Personalized movie and TV show recommendations",
        port=8082,
    )

    print("🚀 Starting Netflix-style streaming demo...")
    demo.run()


def example_4_real_world_corerec_integration():
    """
    Example 4: Realistic integration with CoreRec recommendation models.

    Shows how you might integrate with actual CoreRec models.
    """
    print("🧠 Example 4: CoreRec Model Integration")
    print("=" * 50)

    # Simulate loading a trained CoreRec model
    class MockCorrecModel:
        """Mock CoreRec model for demonstration."""

        def __init__(self):
            self.user_embeddings = {}
            self.item_embeddings = {}
            self.trained = True

        def predict(
                self,
                user_id: str,
                candidate_items: List[str] = None,
                k: int = 10):
            """Mock prediction method."""
            # Simulate real recommendation logic
            time.sleep(0.1)  # Simulate model inference time

            # Generate realistic-looking recommendations
            items = []
            for i in range(k):
                score = random.uniform(0.6, 0.95)
                items.append(
                    {
                        "id": f"track_{user_id}_{i}",
                        "title": f"Recommended Song {i + 1}",
                        "artist": f"Artist {i + 1}",
                        "album": f"Album {i + 1}",
                        "confidence_score": score,
                        "reason": "Based on your listening history",
                    }
                )

            # Sort by confidence score
            items.sort(key=lambda x: x["confidence_score"], reverse=True)
            return items

    # Initialize the model
    model = MockCorrecModel()

    def corerec_recommender(user_id, num_items=12):
        """Wrapper function for CoreRec model."""
        # Call the actual CoreRec model
        recommendations = model.predict(user_id=user_id, k=num_items)

        # The connector will automatically format these for the frontend
        return recommendations

    # Create the demo
    demo = connector(
        predict_function=corerec_recommender,
        frontend="spotify",
        title="CoreRec Music Intelligence",
        description="Advanced music recommendations powered by CoreRec neural collaborative filtering",
        debug=True,  # Enable debug mode to see what's happening
    )

    print("🚀 Starting CoreRec-powered demo...")
    demo.run()


def example_5_background_demo_with_interactions():
    """
    Example 5: Running demo in background and monitoring interactions.

    Shows how to run demos in background and export interaction data.
    """
    print("📊 Example 5: Background Demo with Interaction Monitoring")
    print("=" * 50)

    def recommendation_function(user_id, num_items=10):
        """Simple recommendation function."""
        return [
            {
                "id": f"item_{i}",
                "title": f"Product {i + 1}",
                "category": f"Category {i % 3 + 1}",
                "rating": round(random.uniform(3.5, 5.0), 1),
                "price": f"${random.randint(10, 100)}",
            }
            for i in range(num_items)
        ]

    # Create demo
    demo = connector(
        predict_function=recommendation_function,
        frontend="spotify",  # Using Spotify frontend for product recommendations
        title="E-commerce Recommendations",
        description="Product recommendations in Spotify-style interface",
        auto_open=False,  # Don't auto-open browser
    )

    # Run in background
    print("🚀 Starting demo in background...")
    url = demo.run(background=True)

    print(f"✅ Demo is running at {url}")
    print("📱 Open the URL in your browser and interact with some items")
    print("⏰ Demo will run for 30 seconds, then show interaction data...")

    # Wait and show interactions
    time.sleep(30)

    interactions = demo.get_interactions()
    print(f"\n📊 Recorded {len(interactions)} interactions:")

    for interaction in interactions[-5:]:  # Show last 5 interactions
        print(
            f"   {interaction['user_id']} {interaction['action']} {interaction['item_id']}")

    # Export interactions
    filename = demo.export_interactions()
    print(f"💾 Interactions saved to {filename}")

    # Stop the demo
    demo.stop()


def example_6_multiple_frontends():
    """
    Example 6: Running multiple frontends simultaneously.

    Shows how to run different frontends on different ports.
    """
    print("🔀 Example 6: Multiple Frontend Demos")
    print("=" * 50)

    def music_recommender(user_id, num_items=8):
        return [
            {"id": f"song_{i}", "title": f"Song {i + 1}", "artist": f"Artist {i + 1}"}
            for i in range(num_items)
        ]

    def video_recommender(user_id, num_items=8):
        return [
            {"id": f"video_{i}", "title": f"Video {i + 1}", "channel": f"Channel {i + 1}"}
            for i in range(num_items)
        ]

    def movie_recommender(user_id, num_items=8):
        return [
            {"id": f"movie_{i}", "title": f"Movie {i + 1}", "genre": "Action"}
            for i in range(num_items)
        ]

    # Create multiple demos
    demos = []

    # Spotify demo
    spotify_demo = connector(
        music_recommender,
        "spotify",
        "Music Rec",
        port=8080)
    demos.append(("Spotify", spotify_demo, 8080))

    # YouTube demo
    youtube_demo = connector(
        video_recommender,
        "youtube",
        "Video Rec",
        port=8081)
    demos.append(("YouTube", youtube_demo, 8081))

    # Netflix demo
    netflix_demo = connector(
        movie_recommender,
        "netflix",
        "Movie Rec",
        port=8082)
    demos.append(("Netflix", netflix_demo, 8082))

    # Start all demos in background
    print("🚀 Starting multiple frontend demos...")

    for name, demo, port in demos:
        url = demo.run(background=True, open_browser=False)
        print(f"✅ {name} demo: {url}")

    print("\n🌐 All demos are running! Open these URLs:")
    for name, demo, port in demos:
        print(f"   {name}: http://localhost:{port}")

    print("\n⌨️  Press Enter to stop all demos...")
    input()

    # Stop all demos
    for name, demo, port in demos:
        demo.stop()
        print(f"🛑 Stopped {name} demo")


def run_example(example_number: int):
    """Run a specific example."""
    examples = {
        1: example_1_simple_music_recommender,
        2: example_2_video_recommender_with_different_params,
        3: example_3_netflix_movie_recommender,
        4: example_4_real_world_corerec_integration,
        5: example_5_background_demo_with_interactions,
        6: example_6_multiple_frontends,
    }

    if example_number in examples:
        examples[example_number]()
    else:
        print(f"❌ Example {example_number} not found!")
        print(f"Available examples: {list(examples.keys())}")


def main():
    """Main function to run interactive example selection."""
    print("🎯 CoreRec IMShow Examples")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. Simple Music Recommender (Spotify)")
    print("2. Video Recommender with Different Parameters (YouTube)")
    print("3. Netflix Movie & TV Recommender")
    print("4. Real-world CoreRec Model Integration")
    print("5. Background Demo with Interaction Monitoring")
    print("6. Multiple Frontend Demos")
    print()

    try:
        choice = int(input("Enter example number (1-6): "))
        run_example(choice)
    except (ValueError, KeyboardInterrupt):
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
