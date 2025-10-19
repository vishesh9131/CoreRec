# CoreRec Demo Frontends

Beautiful, platform-specific interfaces for demonstrating recommendation models without writing custom frontends.

## 🌟 Features

- **Pre-built Platform UIs**: Spotify, YouTube, Netflix-inspired interfaces
- **Zero Frontend Code**: Focus on your recommendation algorithms, not UI development
- **Interactive Demos**: Like/dislike, filtering, real-time recommendations
- **Beautiful Design**: Professional, platform-authentic styling
- **Easy Integration**: Drop-in compatibility with CoreRec engines
- **Sample Data**: Built-in data generators for quick demos

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy
```

### 2. Run the Demo

```bash
# Launch platform selection interface
streamlit run corerec/demo_frontends/demo_app.py

# Or launch specific platform directly
streamlit run corerec/demo_frontends/demo_app.py -- --platform spotify
streamlit run corerec/demo_frontends/demo_app.py -- --platform youtube
streamlit run corerec/demo_frontends/demo_app.py -- --platform netflix
```

### 3. Try the Enhanced Interface

```bash
streamlit run corerec/demo_frontends/demo_app.py -- --tabs
```

## 🎯 Available Platforms

### 🎵 Spotify Frontend
- **Theme**: Dark music streaming interface
- **Features**: Track cards, audio features, genre filtering
- **Interactions**: Play, like/dislike, add to playlist
- **Recommendations**: Content-based on audio features and user preferences

### 📺 YouTube Frontend  
- **Theme**: Video platform with YouTube styling
- **Features**: Video cards, categories, trending section
- **Interactions**: Watch, like/dislike, save to watch later
- **Recommendations**: Content-based on video features and viewing history

### 🎬 Netflix Frontend
- **Theme**: Movie/TV streaming interface
- **Features**: Content cards, genre filtering, ratings
- **Interactions**: Watch, rate, add to watchlist
- **Recommendations**: Genre-based and collaborative filtering

## 💻 Usage Examples

### Basic Usage

```python
from corerec.demo_frontends import SpotifyFrontend

# Create and run Spotify demo
frontend = SpotifyFrontend()
frontend.run()
```

### With CoreRec Engine

```python
from corerec.demo_frontends import YouTubeFrontend
from corerec.engines import YourRecommendationEngine

# Initialize your recommendation engine
engine = YourRecommendationEngine()

# Create frontend with engine
frontend = YouTubeFrontend(recommendation_engine=engine)
frontend.run()
```

### With Custom Data

```python
from corerec.demo_frontends import NetflixFrontend

# Use your own dataset
frontend = NetflixFrontend(data_path="your_movies.csv")
frontend.run()
```

### Platform Manager

```python
from corerec.demo_frontends import FrontendManager

# Launch platform selection interface
manager = FrontendManager()
manager.render_platform_selection()

# Or run specific platform
manager.run_platform("spotify")
```

## 📊 Data Formats

### Spotify (Music)
Required columns:
- `track_id`: Unique track identifier
- `track_name`: Track title
- `artist_name`: Artist name
- `genre`: Music genre
- `duration_ms`: Track duration in milliseconds
- `popularity`: Popularity score (0-100)
- `danceability`, `energy`, `valence`: Audio features (0-1)

### YouTube (Videos)
Required columns:
- `video_id`: Unique video identifier
- `title`: Video title
- `channel_name`: Channel name
- `category`: Video category
- `duration_seconds`: Video duration
- `views`: View count
- `likes`: Like count
- `upload_date`: Upload date (YYYY-MM-DD)

### Netflix (Movies/TV)
Required columns:
- `content_id`: Unique content identifier
- `title`: Movie/show title
- `type`: "Movie" or "TV Show"
- `genre`: Content genre
- `release_year`: Release year
- `rating`: Content rating (e.g., "PG-13")
- `imdb_score`: IMDB rating (0-10)

## 🛠️ Extending with New Platforms

### 1. Create Platform Frontend

```python
from corerec.demo_frontends.base_frontend import BaseFrontend
import streamlit as st

class TikTokFrontend(BaseFrontend):
    def __init__(self, recommendation_engine=None, data_path=None):
        theme_config = {
            'primary_color': '#FE2C55',
            'background_color': '#000000',
            'text_color': '#FFFFFF'
        }
        super().__init__("TikTok", recommendation_engine, data_path, theme_config)
    
    def load_data(self):
        # Implement data loading
        pass
    
    def apply_custom_css(self):
        # Implement TikTok styling
        pass
    
    def render_header(self):
        # Implement header
        pass
    
    def render_sidebar(self):
        # Implement controls
        pass
    
    def render_main_content(self):
        # Implement main content area
        pass
    
    def render_item_card(self, item, index=0):
        # Implement content card
        pass
```

### 2. Register Platform

```python
# In frontend_manager.py
self.available_platforms['tiktok'] = {
    'name': 'TikTok',
    'description': 'Short-form video platform',
    'icon': '🎵',
    'class': TikTokFrontend
}
```

## 🎨 Customization

### Theme Configuration

```python
theme_config = {
    'primary_color': '#1DB954',      # Main brand color
    'background_color': '#121212',   # Background
    'text_color': '#FFFFFF',         # Text color
    'secondary_color': '#1ED760',    # Secondary brand color
    'accent_color': '#FF6B6B',       # Accent color
    'card_color': '#181818'          # Card background
}

frontend = SpotifyFrontend(theme_config=theme_config)
```

### Custom CSS

```python
class CustomSpotifyFrontend(SpotifyFrontend):
    def apply_custom_css(self):
        super().apply_custom_css()  # Apply base styles
        
        # Add custom styles
        st.markdown("""
        <style>
        .custom-style {
            /* Your custom CSS */
        }
        </style>
        """, unsafe_allow_html=True)
```

## 🧪 Sample Data Generation

Generate sample datasets for testing:

```python
# Generate Spotify sample data
from corerec.demo_frontends.platforms.spotify_frontend import generate_sample_data
generate_sample_data("spotify_tracks.csv", num_tracks=1000)

# Generate YouTube sample data  
from corerec.demo_frontends.platforms.youtube_frontend import generate_sample_data
generate_sample_data("youtube_videos.csv", num_videos=1000)

# Generate Netflix sample data
from corerec.demo_frontends.platforms.netflix_frontend import generate_sample_data
generate_sample_data("netflix_content.csv", num_items=1000)
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Configure data directory
export COREREC_DATA_DIR=/path/to/your/data

# Optional: Configure cache directory
export COREREC_CACHE_DIR=/path/to/cache
```

### Configuration File

Create `config.yaml`:

```yaml
demo_frontends:
  default_platform: spotify
  sample_data_size: 1000
  enable_caching: true
  
platforms:
  spotify:
    default_genre: all
    audio_features: true
  youtube:
    default_category: all
    trending_enabled: true
  netflix:
    default_type: all
    ratings_enabled: true
```

## 📈 Integration with CoreRec Engines

### Recommendation Engine Interface

Your CoreRec engine should implement:

```python
class YourRecommendationEngine:
    def get_recommendations(self, user_id, num_recommendations=10, filters=None):
        """
        Returns: List of item dictionaries with required fields
        """
        pass
    
    def record_interaction(self, user_id, item_id, interaction_type):
        """
        Record user interaction (like, dislike, play, etc.)
        """
        pass
```

### Example Integration

```python
from corerec.engines import MatrixFactorization
from corerec.demo_frontends import SpotifyFrontend

# Train your model
engine = MatrixFactorization()
engine.fit(training_data)

# Create demo frontend
frontend = SpotifyFrontend(recommendation_engine=engine)
frontend.run()
```

## 🚦 Performance Tips

1. **Sample Data Size**: Use appropriate sample sizes (500-2000 items) for demos
2. **Caching**: Enable caching for large datasets
3. **Filtering**: Implement efficient filtering in your recommendation engine
4. **Lazy Loading**: Load data only when needed

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure CoreRec is in Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/corerec"
   ```

2. **Streamlit Port Conflicts**
   ```bash
   streamlit run demo_app.py --server.port 8502
   ```

3. **Data Loading Issues**
   - Check file paths and permissions
   - Verify CSV column names match requirements
   - Use sample data generators for testing

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

frontend = SpotifyFrontend()
frontend.run()
```

## 📝 License

This project is part of CoreRec and follows the same license terms.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new platforms
4. Submit a pull request

## 📞 Support

- GitHub Issues: Report bugs and request features
- Documentation: See CoreRec main documentation
- Examples: Check the `examples/` directory 