"""
Platform-specific frontend implementations for CoreRec Demo Frontends
"""

from .spotify_frontend import SpotifyFrontend
from .youtube_frontend import YouTubeFrontend
from .netflix_frontend import NetflixFrontend

__all__ = [
    'SpotifyFrontend',
    'YouTubeFrontend', 
    'NetflixFrontend'
] 