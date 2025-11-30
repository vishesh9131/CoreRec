"""
FastAPI Backend for CoreRec Demo Frontends

This module provides REST API endpoints for the demo frontends to interact with
recommendation engines and manage user data.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import uuid
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(title="CoreRec Demo Frontends API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo (in production, use a proper database)
user_sessions = {}
user_interactions = {}


# Models
class UserInteraction(BaseModel):
    user_id: str
    item_id: str
    action: str  # 'like', 'dislike', 'play', 'watch', etc.
    timestamp: Optional[datetime] = None


class RecommendationRequest(BaseModel):
    user_id: str
    platform: str
    num_recommendations: int = 12
    filters: Optional[Dict[str, Any]] = {}


class UserPreferences(BaseModel):
    user_id: str
    platform: str
    categories: List[str] = []
    min_rating: Optional[float] = None
    sort_by: str = "recommended"


# Data loading functions
def load_platform_data(platform: str) -> pd.DataFrame:
    """Load data for a specific platform."""
    try:
        if platform == "spotify":
            from ..platforms.spotify_frontend import SpotifyFrontend

            frontend = SpotifyFrontend()
            df, _ = frontend.load_data()
        elif platform == "youtube":
            from ..platforms.youtube_frontend import YouTubeFrontend

            frontend = YouTubeFrontend()
            df, _ = frontend.load_data()
        elif platform == "netflix":
            from ..platforms.netflix_frontend import NetflixFrontend

            frontend = NetflixFrontend()
            df, _ = frontend.load_data()
        else:
            raise ValueError(f"Unknown platform: {platform}")

        return df
    except Exception as e:
        print(f"Error loading {platform} data: {e}")
        return pd.DataFrame()


# API Endpoints


@app.get("/")
async def root():
    return {"message": "CoreRec Demo Frontends API", "status": "running"}


@app.get("/platforms")
async def get_platforms():
    """Get available platforms."""
    return {
        "platforms": [
            {
                "id": "spotify",
                "name": "Spotify Music Streaming",
                "description": "Music recommendation platform",
                "icon": "🎵",
                "color": "#1DB954",
            },
            {
                "id": "youtube",
                "name": "YouTube Video Platform",
                "description": "Video recommendation platform",
                "icon": "📺",
                "color": "#FF0000",
            },
            {
                "id": "netflix",
                "name": "Netflix Streaming",
                "description": "Movie and TV recommendation platform",
                "icon": "🎬",
                "color": "#E50914",
            },
        ]
    }


@app.get("/platforms/{platform}/data")
async def get_platform_data(platform: str, limit: int = 50):
    """Get sample data for a platform."""
    df = load_platform_data(platform)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for platform: {platform}")

    # Return limited data for performance
    sample_data = df.head(limit).to_dict("records")
    return {"platform": platform, "total_items": len(df), "items": sample_data}


@app.post("/users/{user_id}/interactions")
async def record_interaction(user_id: str, interaction: UserInteraction):
    """Record a user interaction."""
    interaction.user_id = user_id
    interaction.timestamp = datetime.now()

    if user_id not in user_interactions:
        user_interactions[user_id] = []

    user_interactions[user_id].append(interaction.dict())
    return {"status": "success", "message": "Interaction recorded"}


@app.get("/users/{user_id}/interactions")
async def get_user_interactions(user_id: str, platform: Optional[str] = None):
    """Get user interactions."""
    interactions = user_interactions.get(user_id, [])

    if platform:
        # Filter by platform if specified
        interactions = [i for i in interactions if platform in i.get("item_id", "")]

    return {"user_id": user_id, "interactions": interactions}


@app.post("/users/{user_id}/recommendations")
async def get_recommendations(user_id: str, request: RecommendationRequest):
    """Get personalized recommendations for a user."""
    request.user_id = user_id

    # Load platform data
    df = load_platform_data(request.platform)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"No data found for platform: {request.platform}"
        )

    # Get user interactions
    interactions = user_interactions.get(user_id, [])
    liked_items = {i["item_id"] for i in interactions if i["action"] == "like"}
    disliked_items = {i["item_id"] for i in interactions if i["action"] == "dislike"}

    # Apply filters
    filtered_df = df.copy()

    # Remove already interacted items
    all_interacted = liked_items.union(disliked_items)
    if all_interacted:
        id_col = get_id_column(request.platform)
        filtered_df = filtered_df[~filtered_df[id_col].isin(all_interacted)]

    # Apply user filters
    if request.filters:
        filtered_df = apply_filters(filtered_df, request.filters, request.platform)

    # Get recommendations based on user preferences
    if liked_items:
        recommendations = get_content_based_recommendations(
            filtered_df, liked_items, df, request.platform, request.num_recommendations
        )
    else:
        # New user - show popular items
        recommendations = get_popular_items(
            filtered_df, request.platform, request.num_recommendations
        )

    return {
        "user_id": user_id,
        "platform": request.platform,
        "recommendations": recommendations[: request.num_recommendations],
    }


@app.get("/users/{user_id}/trending/{platform}")
async def get_trending(user_id: str, platform: str, limit: int = 6):
    """Get trending items for a platform."""
    df = load_platform_data(platform)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for platform: {platform}")

    trending_items = get_trending_items(df, platform, limit)
    return {"platform": platform, "trending": trending_items}


@app.post("/users/create")
async def create_user():
    """Create a new user session."""
    user_id = str(uuid.uuid4())
    user_sessions[user_id] = {"created_at": datetime.now(), "platforms_used": []}
    return {"user_id": user_id}


# Helper functions


def get_id_column(platform: str) -> str:
    """Get the ID column name for a platform."""
    id_columns = {"spotify": "track_id", "youtube": "video_id", "netflix": "content_id"}
    return id_columns.get(platform, "id")


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any], platform: str) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()

    if platform == "spotify":
        if "genre" in filters and filters["genre"] != "All":
            filtered_df = filtered_df[filtered_df["genre"] == filters["genre"]]
        if "mood" in filters:
            # Filter by valence for mood
            if filters["mood"] == "happy":
                filtered_df = filtered_df[filtered_df["valence"] > 0.6]
            elif filters["mood"] == "sad":
                filtered_df = filtered_df[filtered_df["valence"] < 0.4]

    elif platform == "youtube":
        if "category" in filters and filters["category"] != "All":
            filtered_df = filtered_df[filtered_df["category"] == filters["category"]]
        if "duration" in filters:
            if filters["duration"] == "short":
                filtered_df = filtered_df[filtered_df["duration_seconds"] < 240]
            elif filters["duration"] == "long":
                filtered_df = filtered_df[filtered_df["duration_seconds"] > 1200]

    elif platform == "netflix":
        if "genre" in filters and filters["genre"] != "All":
            filtered_df = filtered_df[filtered_df["genre"] == filters["genre"]]
        if "type" in filters:
            filtered_df = filtered_df[filtered_df["type"].isin(filters["type"])]
        if "min_rating" in filters:
            filtered_df = filtered_df[filtered_df["imdb_rating"] >= filters["min_rating"]]

    return filtered_df


def get_content_based_recommendations(
    df: pd.DataFrame, liked_items: set, full_df: pd.DataFrame, platform: str, num_recs: int
) -> List[Dict]:
    """Get content-based recommendations."""
    id_col = get_id_column(platform)
    liked_df = full_df[full_df[id_col].isin(liked_items)]

    if platform == "spotify":
        # Use audio features for similarity
        feature_cols = ["energy", "valence", "danceability"]
        if all(col in liked_df.columns for col in feature_cols):
            avg_features = liked_df[feature_cols].mean()

            # Calculate similarity scores
            for col in feature_cols:
                df[f"{col}_diff"] = abs(df[col] - avg_features[col])

            df["similarity_score"] = (
                (1 - df["energy_diff"]) * 0.4
                + (1 - df["valence_diff"]) * 0.4
                + (1 - df["danceability_diff"]) * 0.2
            )

            df = df.sort_values("similarity_score", ascending=False)

    elif platform == "youtube":
        # Use video features for similarity
        feature_cols = ["quality_score", "engagement_rate"]
        if all(col in liked_df.columns for col in feature_cols):
            avg_features = liked_df[feature_cols].mean()

            for col in feature_cols:
                df[f"{col}_diff"] = abs(df[col] - avg_features[col])

            df["similarity_score"] = (1 - df["quality_score_diff"]) * 0.5 + (
                1 - df["engagement_rate_diff"]
            ) * 0.5

            df = df.sort_values("similarity_score", ascending=False)

    elif platform == "netflix":
        # Use ratings and genre preferences
        avg_rating = liked_df["imdb_rating"].mean()
        preferred_genres = liked_df["genre"].value_counts()

        df["rating_similarity"] = 1 - abs(df["imdb_rating"] - avg_rating) / 5.0
        df["genre_boost"] = (
            df["genre"].map(lambda x: preferred_genres.get(x, 0) / len(liked_items)).fillna(0)
        )

        df["similarity_score"] = df["rating_similarity"] * 0.6 + df["genre_boost"] * 0.4

        df = df.sort_values("similarity_score", ascending=False)

    return df.head(num_recs).to_dict("records")


def get_popular_items(df: pd.DataFrame, platform: str, num_items: int) -> List[Dict]:
    """Get popular items for new users."""
    if platform == "spotify":
        return df.nlargest(num_items, "popularity").to_dict("records")
    elif platform == "youtube":
        return df.nlargest(num_items, "views").to_dict("records")
    elif platform == "netflix":
        return df.nlargest(num_items, "netflix_score").to_dict("records")
    else:
        return df.head(num_items).to_dict("records")


def get_trending_items(df: pd.DataFrame, platform: str, num_items: int) -> List[Dict]:
    """Get trending items."""
    if platform == "spotify":
        df["trending_score"] = df["popularity"] * 0.6 + df["energy"] * 40
        return df.nlargest(num_items, "trending_score").to_dict("records")
    elif platform == "youtube":
        df["trending_score"] = df["views"] * 0.5 + df["likes"] * 10
        return df.nlargest(num_items, "trending_score").to_dict("records")
    elif platform == "netflix":
        return df.nlargest(num_items, "netflix_score").to_dict("records")
    else:
        return df.head(num_items).to_dict("records")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
