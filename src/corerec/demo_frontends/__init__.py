"""
CoreRec Demo Frontends

Pre-built frontend templates for demonstrating recommendation models
without having to write custom frontends from scratch.

Supported Platforms:
- Spotify-like Music Streaming
- YouTube-like Video Platform
- Netflix-like Movie/TV Streaming
"""

from .base_frontend import BaseFrontend
from .frontend_manager import FrontendManager, quick_launch
from .platforms.spotify_frontend import SpotifyFrontend
from .platforms.youtube_frontend import YouTubeFrontend
from .platforms.netflix_frontend import NetflixFrontend

__version__ = "1.0.0"
__author__ = "CoreRec Team"

__all__ = [
    'BaseFrontend',
    'FrontendManager',
    'quick_launch',
    'SpotifyFrontend',
    'YouTubeFrontend',
    'NetflixFrontend'
] 