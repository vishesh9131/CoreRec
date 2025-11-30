"""
Connector system for plugging user recommendation functions into frontends.
"""

import inspect
import threading
import webbrowser
import time
import uuid
from typing import Callable, Dict, Any, List, Optional, Union
from pathlib import Path
import json

from .frontends import available_frontends, get_frontend_info
from .server import SimpleServer, find_available_port
from .utils import (
    validate_prediction_function,
    format_recommendations,
    parse_function_args,
    generate_user_id,
)


class Connector:
    """
    Main connector class for plugging recommendation functions into frontends.

    Example:
        def my_recommender(user_id, num_items=10):
            # Your recommendation logic
            return [
                {"id": "1", "title": "Song 1", "artist": "Artist 1"},
                {"id": "2", "title": "Song 2", "artist": "Artist 2"},
            ]

        # Plug into Spotify frontend
        connector = Connector(
            predict_function=my_recommender,
            frontend="spotify",
            title="My Music Recommender",
            description="Personalized music recommendations"
        )

        # Run the demo
        connector.run()
    """

    def __init__(
        self,
        predict_function: Callable,
        frontend: str = "spotify",
        title: str = "Recommendation Demo",
        description: str = "Powered by CoreRec",
        port: int = None,
        api_port: int = None,
        debug: bool = False,
        auto_open: bool = True,
        **kwargs,
    ):
        """
        Initialize the connector.

        Args:
            predict_function: User's recommendation function
            frontend: Frontend type ("spotify", "youtube", "netflix")
            title: Demo title
            description: Demo description
            port: Frontend server port (auto-assigned if None)
            api_port: API server port (auto-assigned if None)
            debug: Enable debug mode
            auto_open: Automatically open browser
            **kwargs: Additional configuration
        """
        # Validate inputs
        self.predict_function = predict_function
        validate_prediction_function(predict_function)

        if frontend not in available_frontends():
            raise ValueError(
                f"Frontend '{frontend}' not available. Choose from: {list(available_frontends().keys())}"
            )

        # Store configuration
        self.frontend = frontend
        self.title = title
        self.description = description
        self.debug = debug
        self.auto_open = auto_open
        self.config = kwargs

        # Setup ports
        self.port = port or find_available_port(8080)
        self.api_port = api_port or find_available_port(self.port + 1000)

        # Initialize data storage
        self.interactions = []
        self.user_sessions = {}

        # Create server
        self.server = SimpleServer(self, self.port, self.api_port)

        print(f"🎯 Connector initialized:")
        print(f"   Function: {predict_function.__name__}")
        print(f"   Frontend: {frontend} ({get_frontend_info(frontend)['name']})")
        print(f"   Title: {title}")

    def get_recommendations(
        self, user_id: str, num_items: int = 12, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user by calling the user's function.

        Args:
            user_id: User identifier
            num_items: Number of recommendations to return
            **kwargs: Additional arguments

        Returns:
            List of formatted recommendations
        """
        try:
            # Parse function arguments
            func_args = parse_function_args(
                self.predict_function, user_id=user_id, num_items=num_items, **kwargs, **self.config
            )

            if self.debug:
                print(f"🔍 Calling {self.predict_function.__name__} with args: {func_args}")

            # Call user's prediction function
            raw_recommendations = self.predict_function(**func_args)

            # Format recommendations for frontend
            formatted_recommendations = format_recommendations(raw_recommendations, self.frontend)

            if self.debug:
                print(
                    f"✅ Generated {len(formatted_recommendations)} recommendations for user {user_id}"
                )

            return formatted_recommendations

        except Exception as e:
            if self.debug:
                import traceback

                traceback.print_exc()

            print(f"⚠️ Error getting recommendations: {e}")

            # Return sample data on error
            from .utils import create_sample_data

            return create_sample_data(self.frontend, num_items)

    def record_interaction(self, user_id: str, item_id: str, action: str, **metadata):
        """
        Record user interaction with an item.

        Args:
            user_id: User identifier
            item_id: Item identifier
            action: Action type (play, like, etc.)
            **metadata: Additional interaction data
        """
        interaction = {
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
            "timestamp": time.time(),
            "metadata": metadata,
        }

        self.interactions.append(interaction)

        if self.debug:
            print(f"📝 Recorded interaction: {user_id} {action} {item_id}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get connector information.

        Returns:
            Connector info dictionary
        """
        frontend_info = get_frontend_info(self.frontend)

        return {
            "title": self.title,
            "description": self.description,
            "frontend": {
                "type": self.frontend,
                "name": frontend_info["name"],
                "description": frontend_info["description"],
                "use_case": frontend_info["use_case"],
            },
            "function": {
                "name": self.predict_function.__name__,
                "signature": str(inspect.signature(self.predict_function)),
            },
            "stats": {
                "total_interactions": len(self.interactions),
                "unique_users": len(set(i["user_id"] for i in self.interactions)),
                "total_actions": len(set(i["action"] for i in self.interactions)),
            },
            "urls": {
                "frontend": f"http://localhost:{self.port}",
                "api": f"http://localhost:{self.api_port}",
            },
        }

    def run(self, background: bool = False, open_browser: bool = None):
        """
        Run the demo interface.

        Args:
            background: Run in background (non-blocking)
            open_browser: Whether to open browser (uses auto_open if None)
        """
        if open_browser is None:
            open_browser = self.auto_open

        print(f"\n🚀 Starting {self.title}...")
        print(f"Frontend: {self.frontend} on port {self.port}")
        print(f"API: port {self.api_port}")

        if background:
            # Start in background
            url = self.server.start(background=True)

            if open_browser:
                time.sleep(3)  # Give servers time to start
                print(f"🌐 Opening {url}")
                webbrowser.open(url)

            print(f"✅ Demo running at {url}")
            print("Call connector.stop() to stop the servers")

            return url
        else:
            # Start normally (blocking)
            if open_browser:
                # Open browser in background thread
                def open_browser_delayed():
                    time.sleep(3)
                    url = f"http://localhost:{self.port}"
                    print(f"🌐 Opening {url}")
                    webbrowser.open(url)

                threading.Thread(target=open_browser_delayed, daemon=True).start()

            # This will block
            self.server.start(background=False)

    def stop(self):
        """Stop the demo servers."""
        self.server.stop()

    def get_interactions(self, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Get recorded interactions.

        Args:
            user_id: Filter by user ID (optional)

        Returns:
            List of interactions
        """
        if user_id:
            return [i for i in self.interactions if i["user_id"] == user_id]
        return self.interactions.copy()

    def export_interactions(self, filename: str = None) -> str:
        """
        Export interactions to JSON file.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"corerec_interactions_{timestamp}.json"

        data = {
            "info": self.get_info(),
            "interactions": self.interactions,
            "exported_at": time.time(),
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"💾 Interactions exported to {filename}")
        return filename


def connector(
    predict_function: Callable,
    frontend: str = "spotify",
    title: str = "Recommendation Demo",
    description: str = "Powered by CoreRec",
    **kwargs,
) -> Connector:
    """
    Create a connector instance (convenience function).

    Args:
        predict_function: User's recommendation function
        frontend: Frontend type
        title: Demo title
        description: Demo description
        **kwargs: Additional configuration

    Returns:
        Connector instance
    """
    return Connector(
        predict_function=predict_function,
        frontend=frontend,
        title=title,
        description=description,
        **kwargs,
    )


# Example usage for documentation
def _example_usage():
    """Example usage (for documentation)."""

    # Example 1: Simple music recommender
    def my_music_recommender(user_id, num_items=10):
        """My recommendation function."""
        # Your recommendation logic here
        recommendations = []
        for i in range(num_items):
            recommendations.append(
                {
                    "id": f"song_{i}",
                    "title": f"Great Song {i+1}",
                    "artist": f"Artist {i+1}",
                    "album": f"Album {i+1}",
                }
            )
        return recommendations

    # Create connector
    demo = connector(
        predict_function=my_music_recommender,
        frontend="spotify",
        title="My Music Recommender",
        description="Personalized music recommendations using collaborative filtering",
    )

    # Run the demo
    demo.run()

    # Example 2: Video recommender
    def my_video_recommender(user_id, k=12):
        """Video recommendation function with different parameter name."""
        videos = []
        for i in range(k):
            videos.append(
                {
                    "id": f"video_{i}",
                    "title": f"Amazing Video {i+1}",
                    "channel": f"Channel {i+1}",
                    "duration": f"{3+i}:45",
                }
            )
        return videos

    # YouTube frontend
    video_demo = connector(
        predict_function=my_video_recommender,
        frontend="youtube",
        title="Video Recommendation Engine",
        port=8081,
    )

    # Run in background
    url = video_demo.run(background=True)
    print(f"Demo running at {url}")

    # Export interactions later
    # video_demo.export_interactions("my_interactions.json")


if __name__ == "__main__":
    _example_usage()
