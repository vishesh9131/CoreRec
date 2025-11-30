"""
Frontend configurations and template management for IMShow.
"""

from typing import Dict, Any
from pathlib import Path


def available_frontends() -> Dict[str, Dict[str, Any]]:
    """
    Get available frontend configurations.

    Returns:
        Dictionary of frontend configurations
    """
    return {
        "spotify": {
            "name": "Spotify Music",
            "description": "Music streaming platform with dark theme",
            "theme": "dark_green",
            "use_case": "Music recommendations, audio features, playlists",
            "data_fields": ["title", "artist", "album", "genre", "duration", "popularity"],
            "required_fields": ["id", "title", "artist"],
            "color_scheme": {
                "primary": "#1DB954",
                "background": "#121212",
                "text": "#FFFFFF",
                "secondary": "#535353",
            },
            "template_path": "templates/spotify",
        },
        "youtube": {
            "name": "YouTube Videos",
            "description": "Video platform with red theme",
            "theme": "red_white",
            "use_case": "Video recommendations, content creators, trending",
            "data_fields": ["title", "channel", "category", "duration", "views", "likes"],
            "required_fields": ["id", "title", "channel"],
            "color_scheme": {
                "primary": "#FF0000",
                "background": "#FFFFFF",
                "text": "#0F0F0F",
                "secondary": "#606060",
            },
            "template_path": "templates/youtube",
        },
        "netflix": {
            "name": "Netflix Streaming",
            "description": "Movie and TV show platform with dark theme",
            "theme": "dark_red",
            "use_case": "Movie/TV recommendations, content rating, genres",
            "data_fields": ["title", "genre", "year", "rating", "duration", "description"],
            "required_fields": ["id", "title", "genre"],
            "color_scheme": {
                "primary": "#E50914",
                "background": "#141414",
                "text": "#FFFFFF",
                "secondary": "#564D4D",
            },
            "template_path": "templates/netflix",
        },
    }


def get_frontend_info(frontend: str) -> Dict[str, Any]:
    """
    Get information about a specific frontend.

    Args:
        frontend: Frontend name

    Returns:
        Frontend configuration dictionary
    """
    frontends = available_frontends()
    if frontend not in frontends:
        raise ValueError(
            f"Frontend '{frontend}' not available. Choose from: {
                list(
                    frontends.keys())}")

    return frontends[frontend]


def get_frontend_template(frontend: str) -> str:
    """
    Get the HTML template for a frontend.

    Args:
        frontend: Frontend name

    Returns:
        HTML template string
    """
    frontend_info = get_frontend_info(frontend)

    # Generate template based on frontend type
    if frontend == "spotify":
        return _generate_spotify_template()
    elif frontend == "youtube":
        return _generate_youtube_template()
    elif frontend == "netflix":
        return _generate_netflix_template()
    else:
        return _generate_generic_template(frontend_info)


def _generate_spotify_template() -> str:
    """Generate Spotify-style template."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} - Spotify Demo</title>
    <style>
        body {
            margin: 0;
            font-family: 'Spotify Circular', Helvetica, Arial, sans-serif;
            background: #121212;
            color: #ffffff;
            overflow-x: hidden;
        }
        .container {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 240px;
            background: #000000;
            padding: 20px;
            box-sizing: border-box;
        }
        .logo {
            color: #1DB954;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .main-content {
            flex: 1;
            padding: 20px;
            background: linear-gradient(to bottom, #1e3a5f 0%, #121212 100%);
        }
        .header {
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .track-card {
            background: #181818;
            border-radius: 8px;
            padding: 16px;
            transition: background 0.3s ease;
            cursor: pointer;
        }
        .track-card:hover {
            background: #282828;
        }
        .track-image {
            width: 100%;
            height: 160px;
            background: #1DB954;
            border-radius: 4px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        .track-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .track-artist {
            font-size: 14px;
            color: #b3b3b3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .play-btn {
            background: #1DB954;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .track-card:hover .play-btn {
            opacity: 1;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #b3b3b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="logo">🎵 Spotify</div>
            <div style="color: #b3b3b3; margin-top: 40px;">
                <div style="margin-bottom: 10px;">🏠 Home</div>
                <div style="margin-bottom: 10px;">🔍 Search</div>
                <div style="margin-bottom: 10px;">📚 Your Library</div>
            </div>
        </div>
        <div class="main-content">
            <div class="header">
                <h1>{{title}}</h1>
                <p style="color: #b3b3b3;">{{description}}</p>
            </div>
            <div id="loading" class="loading">
                <div>🎵 Loading your music recommendations...</div>
            </div>
            <div id="recommendations" class="recommendations" style="display: none;">
                <!-- Recommendations will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:{{api_port}}';
        let userId = 'demo_user_' + Math.random().toString(36).substr(2, 9);

        async function loadRecommendations() {
            try {
                const response = await fetch(`${API_URL}/recommendations/${userId}?num_items=12`);
                const data = await response.json();

                displayRecommendations(data.recommendations || []);

                document.getElementById('loading').style.display = 'none';
                document.getElementById('recommendations').style.display = 'grid';
            } catch (error) {
                console.error('Error loading recommendations:', error);
                document.getElementById('loading').innerHTML =
                    '<div style="color: #ff6b6b;">❌ Failed to load recommendations. Make sure the API server is running.</div>';
            }
        }

        function displayRecommendations(recommendations) {
            const container = document.getElementById('recommendations');
            container.innerHTML = '';

            recommendations.forEach(track => {
                const card = document.createElement('div');
                card.className = 'track-card';
                card.innerHTML = `
                    <div class="track-image">🎵</div>
                    <div class="track-title">${track.title || track.name || 'Unknown Track'}</div>
                    <div class="track-artist">${track.artist || track.creator || 'Unknown Artist'}</div>
                    <button class="play-btn" onclick="playTrack('${track.id}')">▶</button>
                `;
                container.appendChild(card);
            });
        }

        async function playTrack(trackId) {
            console.log('Playing track:', trackId);
            // Record interaction
            try {
                await fetch(`${API_URL}/interactions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        item_id: trackId,
                        action: 'play'
                    })
                });
            } catch (error) {
                console.error('Error recording interaction:', error);
            }
        }

        // Load recommendations on page load
        window.addEventListener('load', loadRecommendations);
    </script>
</body>
</html>
    """


def _generate_youtube_template() -> str:
    """Generate YouTube-style template."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} - YouTube Demo</title>
    <style>
        body {
            margin: 0;
            font-family: Roboto, Arial, sans-serif;
            background: #ffffff;
            color: #0f0f0f;
        }
        .header {
            background: #ffffff;
            padding: 0 20px;
            display: flex;
            align-items: center;
            height: 56px;
            border-bottom: 1px solid #e5e5e5;
        }
        .logo {
            color: #ff0000;
            font-size: 20px;
            font-weight: bold;
            margin-right: 40px;
        }
        .container {
            display: flex;
        }
        .sidebar {
            width: 240px;
            background: #ffffff;
            padding: 12px 0;
            border-right: 1px solid #e5e5e5;
            height: calc(100vh - 56px);
            overflow-y: auto;
        }
        .sidebar-item {
            display: flex;
            align-items: center;
            padding: 0 24px;
            height: 40px;
            cursor: pointer;
            color: #030303;
        }
        .sidebar-item:hover {
            background: #f2f2f2;
        }
        .main-content {
            flex: 1;
            padding: 20px;
        }
        .page-title {
            font-size: 24px;
            font-weight: 400;
            margin-bottom: 20px;
        }
        .videos-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        .video-card {
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        .video-card:hover {
            transform: translateY(-2px);
        }
        .video-thumbnail {
            width: 100%;
            height: 180px;
            background: #ff0000;
            border-radius: 8px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 32px;
            position: relative;
        }
        .video-duration {
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 12px;
        }
        .video-info {
            display: flex;
            gap: 12px;
        }
        .channel-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #ff0000;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .video-details {
            flex: 1;
        }
        .video-title {
            font-size: 16px;
            font-weight: 500;
            line-height: 1.3;
            margin-bottom: 4px;
            color: #030303;
        }
        .video-meta {
            font-size: 14px;
            color: #606060;
        }
        .loading {
            text-align: center;
            padding: 60px;
            color: #606060;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">📺 YouTube</div>
        <div style="flex: 1; text-align: center;">
            <strong>{{title}}</strong>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="sidebar-item">🏠 Home</div>
            <div class="sidebar-item">🔥 Trending</div>
            <div class="sidebar-item">📚 Subscriptions</div>
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #e5e5e5;">
            <div class="sidebar-item">📺 Library</div>
            <div class="sidebar-item">🕒 History</div>
            <div class="sidebar-item">👍 Liked videos</div>
        </div>

        <div class="main-content">
            <div class="page-title">Recommended for you</div>
            <div id="loading" class="loading">
                <div>📺 Loading your video recommendations...</div>
            </div>
            <div id="videos" class="videos-grid" style="display: none;">
                <!-- Videos will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:{{api_port}}';
        let userId = 'demo_user_' + Math.random().toString(36).substr(2, 9);

        async function loadRecommendations() {
            try {
                const response = await fetch(`${API_URL}/recommendations/${userId}?num_items=12`);
                const data = await response.json();

                displayVideos(data.recommendations || []);

                document.getElementById('loading').style.display = 'none';
                document.getElementById('videos').style.display = 'grid';
            } catch (error) {
                console.error('Error loading recommendations:', error);
                document.getElementById('loading').innerHTML =
                    '<div style="color: #ff0000;">❌ Failed to load recommendations. Make sure the API server is running.</div>';
            }
        }

        function displayVideos(videos) {
            const container = document.getElementById('videos');
            container.innerHTML = '';

            videos.forEach(video => {
                const card = document.createElement('div');
                card.className = 'video-card';
                card.onclick = () => watchVideo(video.id);

                card.innerHTML = `
                    <div class="video-thumbnail">
                        📺
                        <div class="video-duration">${video.duration || '5:23'}</div>
                    </div>
                    <div class="video-info">
                        <div class="channel-avatar">📺</div>
                        <div class="video-details">
                            <div class="video-title">${video.title || video.name || 'Unknown Video'}</div>
                            <div class="video-meta">
                                <div>${video.channel || video.creator || 'Unknown Channel'}</div>
                                <div>${video.views || Math.floor(Math.random() * 1000000).toLocaleString()} views • ${video.uploaded || '2 days ago'}</div>
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        async function watchVideo(videoId) {
            console.log('Watching video:', videoId);
            // Record interaction
            try {
                await fetch(`${API_URL}/interactions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        item_id: videoId,
                        action: 'watch'
                    })
                });
            } catch (error) {
                console.error('Error recording interaction:', error);
            }
        }

        // Load recommendations on page load
        window.addEventListener('load', loadRecommendations);
    </script>
</body>
</html>
    """


def _generate_netflix_template() -> str:
    """Generate Netflix-style template."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}} - Netflix Demo</title>
    <style>
        body {
            margin: 0;
            font-family: 'Netflix Sans', Helvetica, Arial, sans-serif;
            background: #141414;
            color: #ffffff;
        }
        .header {
            background: #141414;
            padding: 0 60px;
            display: flex;
            align-items: center;
            height: 68px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            border-bottom: 1px solid #333;
        }
        .logo {
            color: #e50914;
            font-size: 24px;
            font-weight: bold;
            margin-right: 40px;
        }
        .nav-menu {
            display: flex;
            gap: 20px;
            flex: 1;
        }
        .nav-item {
            color: #e5e5e5;
            text-decoration: none;
            font-size: 14px;
            padding: 4px 0;
            cursor: pointer;
        }
        .nav-item:hover, .nav-item.active {
            color: #ffffff;
        }
        .main-content {
            margin-top: 68px;
            padding: 40px 60px;
        }
        .hero-section {
            margin-bottom: 40px;
        }
        .hero-title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 16px;
        }
        .hero-description {
            font-size: 18px;
            color: #b3b3b3;
            max-width: 600px;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 16px;
            color: #e5e5e5;
        }
        .content-row {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 8px;
            margin-bottom: 40px;
        }
        .content-card {
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.3s ease;
            position: relative;
        }
        .content-card:hover {
            transform: scale(1.05);
            z-index: 10;
        }
        .content-image {
            width: 100%;
            height: 140px;
            background: #e50914;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }
        .content-info {
            padding: 12px;
        }
        .content-title {
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .content-meta {
            font-size: 12px;
            color: #b3b3b3;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rating {
            color: #46d369;
            font-weight: bold;
        }
        .loading {
            text-align: center;
            padding: 80px;
            color: #b3b3b3;
        }
        @media (max-width: 768px) {
            .header {
                padding: 0 20px;
            }
            .main-content {
                padding: 20px;
            }
            .content-row {
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🎬 Netflix</div>
        <div class="nav-menu">
            <div class="nav-item active">Home</div>
            <div class="nav-item">TV Shows</div>
            <div class="nav-item">Movies</div>
            <div class="nav-item">New & Popular</div>
            <div class="nav-item">My List</div>
        </div>
    </div>

    <div class="main-content">
        <div class="hero-section">
            <div class="hero-title">{{title}}</div>
            <div class="hero-description">{{description}}</div>
        </div>

        <div id="loading" class="loading">
            <div>🎬 Loading your personalized recommendations...</div>
        </div>

        <div id="content-sections" style="display: none;">
            <div class="section-title">Recommended for You</div>
            <div id="recommendations" class="content-row">
                <!-- Content will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:{{api_port}}';
        let userId = 'demo_user_' + Math.random().toString(36).substr(2, 9);

        async function loadRecommendations() {
            try {
                const response = await fetch(`${API_URL}/recommendations/${userId}?num_items=12`);
                const data = await response.json();

                displayContent(data.recommendations || []);

                document.getElementById('loading').style.display = 'none';
                document.getElementById('content-sections').style.display = 'block';
            } catch (error) {
                console.error('Error loading recommendations:', error);
                document.getElementById('loading').innerHTML =
                    '<div style="color: #e50914;">❌ Failed to load recommendations. Make sure the API server is running.</div>';
            }
        }

        function displayContent(content) {
            const container = document.getElementById('recommendations');
            container.innerHTML = '';

            content.forEach(item => {
                const card = document.createElement('div');
                card.className = 'content-card';
                card.onclick = () => watchContent(item.id);

                card.innerHTML = `
                    <div class="content-image">🎬</div>
                    <div class="content-info">
                        <div class="content-title">${item.title || item.name || 'Unknown Title'}</div>
                        <div class="content-meta">
                            <span>${item.genre || item.category || 'Drama'}</span>
                            <span class="rating">${item.rating || (Math.random() * 2 + 8).toFixed(1)}</span>
                        </div>
                    </div>
                `;
                container.appendChild(card);
            });
        }

        async function watchContent(contentId) {
            console.log('Watching content:', contentId);
            // Record interaction
            try {
                await fetch(`${API_URL}/interactions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        item_id: contentId,
                        action: 'watch'
                    })
                });
            } catch (error) {
                console.error('Error recording interaction:', error);
            }
        }

        // Load recommendations on page load
        window.addEventListener('load', loadRecommendations);
    </script>
</body>
</html>
    """


def _generate_generic_template(frontend_info: Dict[str, Any]) -> str:
    """Generate a generic template for unknown frontends."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 16px;
        }
        .description {
            font-size: 18px;
            color: #666;
        }
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .item-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
            cursor: pointer;
        }
        .item-card:hover {
            transform: translateY(-2px);
        }
        .item-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .item-meta {
            color: #666;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            padding: 60px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">{{title}}</div>
            <div class="description">{{description}}</div>
        </div>

        <div id="loading" class="loading">
            <div>📋 Loading recommendations...</div>
        </div>

        <div id="recommendations" class="recommendations" style="display: none;">
            <!-- Recommendations will be populated by JavaScript -->
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:{{api_port}}';
        let userId = 'demo_user_' + Math.random().toString(36).substr(2, 9);

        async function loadRecommendations() {
            try {
                const response = await fetch(`${API_URL}/recommendations/${userId}?num_items=12`);
                const data = await response.json();

                displayRecommendations(data.recommendations || []);

                document.getElementById('loading').style.display = 'none';
                document.getElementById('recommendations').style.display = 'grid';
            } catch (error) {
                console.error('Error loading recommendations:', error);
                document.getElementById('loading').innerHTML =
                    '<div style="color: #e74c3c;">❌ Failed to load recommendations. Make sure the API server is running.</div>';
            }
        }

        function displayRecommendations(recommendations) {
            const container = document.getElementById('recommendations');
            container.innerHTML = '';

            recommendations.forEach(item => {
                const card = document.createElement('div');
                card.className = 'item-card';
                card.onclick = () => selectItem(item.id);

                card.innerHTML = `
                    <div class="item-title">${item.title || item.name || 'Unknown Item'}</div>
                    <div class="item-meta">
                        ${Object.entries(item).filter(([k,v]) => k !== 'id' && k !== 'title' && k !== 'name').map(([k,v]) => `${k}: ${v}`).join(' • ')}
                    </div>
                `;
                container.appendChild(card);
            });
        }

        async function selectItem(itemId) {
            console.log('Selected item:', itemId);
            // Record interaction
            try {
                await fetch(`${API_URL}/interactions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        item_id: itemId,
                        action: 'select'
                    })
                });
            } catch (error) {
                console.error('Error recording interaction:', error);
            }
        }

        // Load recommendations on page load
        window.addEventListener('load', loadRecommendations);
    </script>
</body>
</html>
    """
