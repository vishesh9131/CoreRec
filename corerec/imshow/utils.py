"""
Utility functions for the IMShow connector system.
"""

import inspect
import random
import string
from typing import Callable, List, Dict, Any, Union


def validate_prediction_function(func: Callable) -> bool:
    """
    Validate that a prediction function has the correct signature and is callable.

    Args:
        func: Function to validate

    Returns:
        True if valid

    Raises:
        ValueError: If function is not valid
    """
    if not callable(func):
        raise ValueError("Prediction function must be callable")

    # Check function signature
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Function should accept some parameters (user_id, num_items, etc.)
        # or no parameters at all for simple cases

        print(f"✅ Function '{func.__name__}' signature: {list(params)}")
        return True

    except Exception as e:
        raise ValueError(f"Error inspecting function signature: {e}")


def format_recommendations(
    recommendations: Union[List[Dict], List[Any]], frontend: str
) -> List[Dict[str, Any]]:
    """
    Format recommendations to match frontend requirements.

    Args:
        recommendations: Raw recommendations from user function
        frontend: Frontend type

    Returns:
        Formatted recommendations list
    """
    if not recommendations:
        return create_sample_data(frontend, 12)

    formatted = []

    for i, rec in enumerate(recommendations):
        if isinstance(rec, dict):
            # Already a dictionary
            formatted_rec = dict(rec)
        else:
            # Convert to dictionary
            formatted_rec = {"title": str(rec), "id": f"item_{i}"}

        # Ensure required fields exist
        if "id" not in formatted_rec:
            formatted_rec["id"] = f"item_{i}"

        # Add frontend-specific fields
        if frontend == "spotify":
            if "title" not in formatted_rec and "name" not in formatted_rec:
                formatted_rec["title"] = f"Track {i + 1}"
            if "artist" not in formatted_rec:
                formatted_rec["artist"] = f"Artist {i + 1}"
            if "album" not in formatted_rec:
                formatted_rec["album"] = f"Album {i + 1}"
            if "duration" not in formatted_rec:
                formatted_rec["duration"] = f"{random.randint(180, 300)}s"

        elif frontend == "youtube":
            if "title" not in formatted_rec and "name" not in formatted_rec:
                formatted_rec["title"] = f"Video {i + 1}"
            if "channel" not in formatted_rec:
                formatted_rec["channel"] = f"Channel {i + 1}"
            if "views" not in formatted_rec:
                formatted_rec["views"] = f"{random.randint(1000, 1000000):,}"
            if "duration" not in formatted_rec:
                minutes = random.randint(2, 30)
                seconds = random.randint(0, 59)
                formatted_rec["duration"] = f"{minutes}:{seconds:02d}"

        elif frontend == "netflix":
            if "title" not in formatted_rec and "name" not in formatted_rec:
                formatted_rec["title"] = f"Movie {i + 1}"
            if "genre" not in formatted_rec:
                genres = [
                    "Drama",
                    "Comedy",
                    "Action",
                    "Thriller",
                    "Romance",
                    "Sci-Fi",
                    "Horror"]
                formatted_rec["genre"] = random.choice(genres)
            if "rating" not in formatted_rec:
                formatted_rec["rating"] = f"{random.uniform(7.0, 9.5):.1f}"
            if "year" not in formatted_rec:
                formatted_rec["year"] = random.randint(2015, 2024)

        formatted.append(formatted_rec)

    return formatted


def create_sample_data(
        frontend: str, num_items: int = 12) -> List[Dict[str, Any]]:
    """
    Create sample data for a frontend when no recommendations are provided.

    Args:
        frontend: Frontend type
        num_items: Number of sample items to create

    Returns:
        List of sample recommendation dictionaries
    """
    sample_data = []

    if frontend == "spotify":
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
        genres = [
            "Pop",
            "Hip-Hop",
            "Rock",
            "Electronic",
            "R&B",
            "Indie",
            "Jazz",
            "Country"]

        for i in range(num_items):
            sample_data.append(
                {
                    "id": f"track_{i + 1}",
                    "title": f"Sample Song {i + 1}",
                    "artist": random.choice(artists),
                    "album": f"Album {i + 1}",
                    "genre": random.choice(genres),
                    "duration": f"{random.randint(180, 300)}s",
                    "popularity": random.randint(60, 100),
                }
            )

    elif frontend == "youtube":
        channels = [
            "TechReview",
            "MusicHub",
            "GameZone",
            "CookingMaster",
            "TravelVlog",
            "ScienceExplained",
            "DIYProjects",
            "FitnessGuru",
            "NewsToday",
            "Comedy Central",
        ]
        categories = [
            "Technology",
            "Music",
            "Gaming",
            "Food",
            "Travel",
            "Science",
            "DIY",
            "Fitness",
            "News",
            "Comedy",
        ]

        for i in range(num_items):
            views = random.randint(1000, 5000000)
            minutes = random.randint(2, 30)
            seconds = random.randint(0, 59)

            sample_data.append(
                {
                    "id": f"video_{
                        i + 1}",
                    "title": f"Amazing Video {
                        i + 1} - You Won't Believe This!",
                    "channel": random.choice(channels),
                    "category": random.choice(categories),
                    "views": f"{
                        views:,                }",
                    "duration": f"{minutes}:{
                        seconds:02d}",
                    "uploaded": f"{
                        random.randint(
                            1,
                            30)} days ago",
                    "likes": f"{
                        random.randint(
                            100,
                            50000):,            }",
                })

    elif frontend == "netflix":
        genres = [
            "Drama",
            "Comedy",
            "Action",
            "Thriller",
            "Romance",
            "Sci-Fi",
            "Horror",
            "Documentary",
            "Family",
            "Crime",
            "Fantasy",
            "Mystery",
        ]
        types = ["Movie", "TV Show", "Documentary", "Limited Series"]

        for i in range(num_items):
            sample_data.append(
                {
                    "id": f"content_{i + 1}",
                    "title": f"Sample {random.choice(types)} {i + 1}",
                    "genre": random.choice(genres),
                    "year": random.randint(2015, 2024),
                    "rating": f"{random.uniform(7.0, 9.5):.1f}",
                    "duration": f"{random.randint(90, 180)} min"
                    if random.choice(types) == "Movie"
                    else f"{random.randint(1, 8)} Seasons",
                    "type": random.choice(types),
                    "description": f"An engaging {random.choice(genres).lower()} story that will keep you entertained.",
                }
            )

    else:
        # Generic sample data
        for i in range(num_items):
            sample_data.append(
                {
                    "id": f"item_{i + 1}",
                    "title": f"Sample Item {i + 1}",
                    "category": f"Category {(i % 5) + 1}",
                    "score": round(random.uniform(0.7, 1.0), 2),
                    "description": f"This is a sample recommendation item for demonstration purposes.",
                }
            )

    return sample_data


def generate_user_id() -> str:
    """Generate a random user ID."""
    return f"user_{
        ''.join(
            random.choices(
                string.ascii_lowercase +
                string.digits,
                k=8))}"


def parse_function_args(func: Callable, **kwargs) -> Dict[str, Any]:
    """
    Parse and prepare arguments for calling a user's prediction function.

    Args:
        func: User's prediction function
        **kwargs: Arguments to potentially pass

    Returns:
        Dictionary of arguments that match the function signature
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Prepare arguments based on function signature
    args = {}

    # Common parameter mappings
    param_mappings = {
        "user_id": ["user_id", "user", "user_identifier"],
        "num_items": ["num_items", "n_items", "k", "top_k", "limit"],
        "num_recommendations": ["num_recommendations", "n_recommendations"],
        "item_id": ["item_id", "item", "product_id"],
        "user_features": ["user_features", "user_data"],
        "item_features": ["item_features", "item_data"],
    }

    # Match kwargs to function parameters
    for param in params:
        if param in kwargs:
            args[param] = kwargs[param]
        else:
            # Try to find a match in mappings
            for key, aliases in param_mappings.items():
                if param in aliases and key in kwargs:
                    args[param] = kwargs[key]
                    break

    return args


def validate_frontend_data(data: List[Dict], frontend: str) -> bool:
    """
    Validate that data has required fields for a frontend.

    Args:
        data: List of data items
        frontend: Frontend type

    Returns:
        True if valid
    """
    if not data:
        return False

    from .frontends import get_frontend_info

    try:
        frontend_info = get_frontend_info(frontend)
        required_fields = frontend_info.get("required_fields", ["id", "title"])

        for item in data[:5]:  # Check first 5 items
            if not isinstance(item, dict):
                return False

            for field in required_fields:
                if field not in item:
                    return False

        return True

    except Exception:
        return False


def format_duration(seconds: Union[int, float, str]) -> str:
    """
    Format duration in different ways based on input.

    Args:
        seconds: Duration in seconds or string format

    Returns:
        Formatted duration string
    """
    if isinstance(seconds, str):
        return seconds

    try:
        total_seconds = int(float(seconds))
        minutes = total_seconds // 60
        secs = total_seconds % 60

        if total_seconds < 3600:  # Less than an hour
            return f"{minutes}:{secs:02d}"
        else:  # More than an hour
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"

    except (ValueError, TypeError):
        return str(seconds)


def clean_text(text: str, max_length: int = 100) -> str:
    """
    Clean and truncate text for display.

    Args:
        text: Text to clean
        max_length: Maximum length

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove extra whitespace
    text = " ".join(text.split())

    # Truncate if too long
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text


def get_mock_interaction_data(
        user_id: str, num_interactions: int = 10) -> List[Dict[str, Any]]:
    """
    Generate mock interaction data for testing.

    Args:
        user_id: User identifier
        num_interactions: Number of interactions to generate

    Returns:
        List of mock interactions
    """
    actions = ["play", "like", "dislike", "skip", "share", "watch", "rate"]
    interactions = []

    for i in range(num_interactions):
        interactions.append(
            {
                "user_id": user_id, "item_id": f"item_{
                    random.randint(
                        1, 100)}", "action": random.choice(actions), "timestamp": f"2024-01-{
                    random.randint(
                        1, 30):02d}T{
                            random.randint(
                                0, 23):02d}:{
                                    random.randint(
                                        0, 59):02d}:00", "score": random.uniform(
                                            0.1, 1.0), })

    return interactions
